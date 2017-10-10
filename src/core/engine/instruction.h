#ifndef instruction_h
#define instruction_h

#include <vector>

#include "network/connection.h"
#include "state/state.h"
#include "state/attributes.h"
#include "io/buffer.h"
#include "engine/engine.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/attribute_data.h"

class Instruction {
    public:
        Instruction(Layer *layer, Stream *stream)
                : to_layer(layer),
                  threads(calc_threads(layer->size)),
                  blocks(calc_blocks(layer->size)),
                  stream(stream),
                  event(nullptr),
                  child(nullptr) { }
        virtual ~Instruction() { }

        virtual void activate() = 0;

        void wait_for_dependencies() {
            // If this instruction has a child, activate it first
            if (child != nullptr) child->activate();
            for (auto& dep : dependencies) stream->wait(dep);
        }

        void add_event() {
            if (event == nullptr)
                event = ResourceManager::get_instance()->create_event(
                    stream->get_device_id());
        }
        void record_event() { if (event != nullptr) stream->record(event); }
        void synchronize() { if (event != nullptr) event->wait(nullptr); }

        void add_dependency(Instruction *inst) {
            inst->add_event();
            this->dependencies.push_back(inst->event);
        }

        void copy_dependencies(Instruction *inst) {
            for (auto dep : inst->dependencies)
                this->dependencies.push_back(dep);
        }

        /* Adds a child instruction which will be completed before this
         *   instruction is activated.  Dependencies are transferred to
         *   the child, and replaced with the child. */
        void add_child(Instruction *inst) {
            this->child = inst;
            inst->copy_dependencies(this);
            this->dependencies.clear();
            this->add_dependency(inst);
        }

        Layer* const to_layer;

    protected:
        Stream *stream;
        Event* event;
        std::vector<Event*> dependencies;
        int blocks, threads;
        Instruction *child;
};

/* Inter-device connections transfer instruction */
class InterDeviceTransferInstruction : public Instruction {
    public:
        InterDeviceTransferInstruction(
            Connection *conn, State *state)
                : Instruction(conn->to_layer, nullptr) {
            if (not state->is_inter_device(conn))
                LOG_ERROR(
                    "Error creating instruction for " + conn->str() + ":\n"
                    "  InterDeviceTransferInstruction should only be used with"
                    " inter-device synaptic connections!");

            int word_index = get_word_index(
                conn->delay, Attributes::get_output_type(conn->from_layer));
            src = state->get_output(conn->from_layer, word_index);
            dst = state->get_device_output_buffer(conn, word_index);

            DeviceID source_device = state->get_device_id(conn->from_layer);
            stream = ResourceManager::get_instance()
                ->get_inter_device_stream(source_device);
            event = ResourceManager::get_instance()
                ->create_event(source_device);
        }

        void activate() {
            Instruction::wait_for_dependencies();
            src.copy_to(dst, stream);
            Instruction::record_event();
        }

        bool matches(Connection *conn, State *state) {
            int word_index = get_word_index(
                conn->delay, Attributes::get_output_type(conn->from_layer));
            auto other_src = state->get_output(conn->from_layer, word_index);
            auto other_dst = state->get_device_output_buffer(conn, word_index);

            return this->src == other_src and this->dst == other_dst;
        }

    protected:
        Pointer<Output> src, dst;
};

/* Instructions that initialize the input without connections */
class InitializeInstruction : public Instruction {
    public:
        // Initialize layer buffers
        InitializeInstruction(Layer *layer, State *state, Stream *stream)
                : Instruction(layer, stream),
                  dst(state->get_input(layer)),
                  size(layer->size) { }

        // Initialize second order buffers
        InitializeInstruction(DendriticNode *second_order_node,
            State *state, Stream *stream)
                : Instruction(second_order_node->to_layer, stream),
                  dst(state->get_second_order_weights(second_order_node)),
                  size(second_order_node->get_second_order_size()) { }

    protected:
        Pointer<float> dst;
        int size;
};

/* Clears inputs */
class ClearInstruction : public InitializeInstruction {
    public:
        // Clear layer buffers
        ClearInstruction(Layer *layer, State *state, Stream *stream)
                : InitializeInstruction(layer, state, stream) { }

        void activate() {
            wait_for_dependencies();
            get_set_data().run(stream,
                blocks, threads,
                0.0, dst, size);
            Instruction::record_event();
        }
};

/* Adds noise to the input */
class NormalNoiseInstruction : public InitializeInstruction {
    public:
        NormalNoiseInstruction(Layer *layer, State *state,
            Stream *stream, bool init)
                : InitializeInstruction(layer, state, stream),
                  init(init),
                  mean(layer->get_config()->get_child("noise config")
                      ->get_float("mean", 1.0)),
                  std_dev(layer->get_config()->get_child("noise config")
                      ->get_float("std_dev", 0.1)) { }

        void activate() {
            Instruction::wait_for_dependencies();
            get_randomize_data_normal().run(stream,
                blocks, threads,
                dst, size,
                mean, std_dev,
                init);
            Instruction::record_event();
        }

    protected:
        float mean;
        float std_dev;
        bool init;
};

class PoissonNoiseInstruction : public InitializeInstruction {
    public:
        PoissonNoiseInstruction(Layer *layer, State *state,
            Stream *stream, bool init)
                : InitializeInstruction(layer, state, stream),
                  init(init),
                  val(layer->get_config()->get_child("noise config")
                      ->get_float("value", 20)),
                  rate(0.001 * layer->get_config()->get_child("noise config")
                      ->get_float("rate", 1)) { }

        void activate() {
            Instruction::wait_for_dependencies();
            get_randomize_data_poisson().run(stream,
                blocks, threads,
                dst, size,
                val, rate,
                init);
            Instruction::record_event();
        }

    protected:
        float val;
        float rate;
        bool init;
};

/* Operates on synapses */
class SynapseInstruction : public Instruction {
    public:
        SynapseInstruction(DendriticNode* parent_node, Connection *conn,
            State *state, Stream *stream)
                : Instruction(conn->to_layer, stream),
                  connection(conn),
                  synapse_data(parent_node, conn, state),
                  type(conn->type) { }

        const ConnectionType type;
        Connection* const connection;

    protected:
        SynapseData synapse_data;
};

/* Activates synaptic connection */
class SynapseActivateInstruction : public SynapseInstruction {
    public:
        SynapseActivateInstruction(DendriticNode *parent_node,
            Connection *conn, State *state, Stream *stream)
                : SynapseInstruction(parent_node, conn, state, stream),
                  activator(state->get_activator(conn)) {
            // Convolutional activate instructions iterate over weights
            // This is because of special conditions (see connection.cpp)
            if (conn->second_order_slave and conn->type == CONVOLUTIONAL) {
                int num_weights = connection->get_num_weights();
                this->threads = calc_threads(num_weights);
                this->blocks = calc_blocks(num_weights);
            }
        }

        void activate() {
            Instruction::wait_for_dependencies();
            activator.run(stream,
                blocks, threads,
                synapse_data);
            Instruction::record_event();
        }

    protected:
        Kernel<SYNAPSE_ARGS> activator;
};

/* Updates synaptic connection */
class SynapseUpdateInstruction : public SynapseInstruction {
    public:
        SynapseUpdateInstruction(DendriticNode *parent_node,
            Connection *conn, State *state, Stream *stream)
                : SynapseInstruction(parent_node, conn, state, stream),
                  updater(state->get_updater(conn)) {
            // Convolutional update instructions iterate over weights
            if (conn->type == CONVOLUTIONAL) {
                int num_weights = connection->get_num_weights();
                this->threads = calc_threads(num_weights);
                this->blocks = calc_blocks(num_weights);
            }
        }

        void activate() {
            updater.run(stream,
                blocks, threads,
                synapse_data);
        }

    protected:
        Kernel<SYNAPSE_ARGS> updater;
};

/* Computes dendritic node connection */
class DendriticInstruction : public Instruction {
    public:
        DendriticInstruction(DendriticNode *parent,
            DendriticNode *child, State *state, Stream *stream)
                : Instruction(parent->to_layer, stream),
                  init(child->register_index != 0),
                  src(state->get_input(to_layer, child->register_index)),
                  dst(state->get_input(to_layer, parent->register_index)) { }

        void activate() {
            Instruction::wait_for_dependencies();
            get_calc_internal().run(
                stream, blocks, threads,
                to_layer->size, src, dst, init);
            Instruction::record_event();
        }

    protected:
        Pointer<float> src, dst;
        bool init;
};

/* Transfers data */
template<class T>
class TransferInstruction : public Instruction {
    public:
        TransferInstruction(Layer *layer, Stream *stream,
            Pointer<T> src, Pointer<T> dst)
                : Instruction(layer, stream),
                  src(src), dst(dst) {
            this->add_event();
        }

        void activate() {
            Instruction::wait_for_dependencies();
            src.copy_to(dst, stream);
            Instruction::record_event();
        }

    protected:
        Pointer<T> src, dst;
};

/* Transfers data with an intermediate buffer */
template<class T>
class BufferedTransferInstruction : public Instruction {
    public:
        BufferedTransferInstruction(Layer *layer, Stream *stream,
            Pointer<T> src, Pointer<T> inter, Pointer<T> dst,
            Buffer *buffer, bool check_dirty=false)
                : Instruction(layer, stream),
                  src(src), inter(inter), dst(dst),
                  buffer(buffer) {
            this->add_event();
        }

        void activate() {
            Instruction::wait_for_dependencies();

            if (not check_dirty or buffer->get_dirty(to_layer)) {
                buffer->set_dirty(to_layer, false);
                src.copy_to(inter, stream);
            }
            inter.copy_to(dst, stream);

            Instruction::record_event();
        }

    protected:
        Pointer<T> src, inter, dst;
        Buffer* buffer;
        bool check_dirty;
};

/* Transfers weights intradevice for second order connections */
class SecondOrderWeightTransferInstruction : public TransferInstruction<float> {
    public:
        SecondOrderWeightTransferInstruction(DendriticNode *node,
            State *state, Stream *stream)
                : TransferInstruction(
                      node->get_second_order_connection()->to_layer,
                      stream,
                      state->get_matrix(node->get_second_order_connection()),
                      state->get_second_order_weights(node)) { }
};

/* Transfers input data */
/* Sets input from buffer */
class InputTransferInstruction : public BufferedTransferInstruction<float> {
    public:
        InputTransferInstruction(Layer *layer, State *state,
            Engine *engine, Stream *stream)
                : BufferedTransferInstruction(layer, stream,
                      engine->get_buffer()->get_input(layer),
                      state->get_buffer_input(layer),
                      state->get_input(layer),
                      engine->get_buffer()) { }
};

/* Transfers expected data */
class ExpectedTransferInstruction : public BufferedTransferInstruction<Output> {
    public:
        ExpectedTransferInstruction(Layer *layer, State *state,
            Engine *engine, Stream *stream)
                : BufferedTransferInstruction(layer, stream,
                      engine->get_buffer()->get_expected(layer),
                      state->get_buffer_expected(layer),
                      state->get_expected(layer),
                      engine->get_buffer()) { }
};

/* Transfers output data */
class OutputTransferInstruction : public TransferInstruction<Output> {
    public:
        OutputTransferInstruction(Layer *layer, State *state,
            Engine *engine, Stream *stream)
                : TransferInstruction(layer, stream,
                      state->get_output(layer),
                      engine->get_buffer()->get_output(layer)) { }
};

/* Operates on neuron state */
class StateInstruction : public Instruction {
    public:
        StateInstruction(Layer *to_layer, State *state, Stream *stream,
            Kernel<ATTRIBUTE_ARGS> attribute_kernel)
            : Instruction(to_layer, stream),
              attribute_data(to_layer, state),
              attribute_kernel(attribute_kernel) { }

        void activate() {
            Instruction::wait_for_dependencies();
            attribute_kernel.run(stream,
                blocks, threads,
                attribute_data);
            Instruction::record_event();
        }

    protected:
        AttributeData attribute_data;
        Kernel<ATTRIBUTE_ARGS> attribute_kernel;
};

/* Updates layer state */
class StateUpdateInstruction : public StateInstruction {
    public:
        StateUpdateInstruction(Layer *to_layer, State *state, Stream *stream)
            : StateInstruction(to_layer, state, stream,
              state->get_attribute_kernel(to_layer)) { }
};

/* Updates layer state */
class StateLearningInstruction : public StateInstruction {
    public:
        StateLearningInstruction(Layer *to_layer, State *state, Stream *stream)
            : StateInstruction(to_layer, state, stream,
              state->get_learning_kernel(to_layer)) { }
};

typedef std::vector<Instruction*> InstructionList;
typedef std::vector<SynapseInstruction*> SynapseInstructionList;

#endif
