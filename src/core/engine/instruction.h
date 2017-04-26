#ifndef instruction_h
#define instruction_h

#include <vector>

#include "model/connection.h"
#include "state/state.h"
#include "state/attributes.h"
#include "io/buffer.h"
#include "io/environment.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/attribute_data.h"

class Instruction {
    public:
        Instruction(Layer *layer, Stream *stream)
                : to_layer(layer),
                  threads(calc_threads(layer->size)),
                  blocks(calc_blocks(layer->size)),
                  stream(stream),
                  event(nullptr) { }
        virtual ~Instruction() { }

        virtual void activate() = 0;

        void wait_for_dependencies() {
            for (auto& dep : dependencies) stream->wait(dep);
        }

        void add_event() {
            if (event == nullptr)
                event = ResourceManager::get_instance()->create_event(
                    stream->get_device_id());
        }
        void record_event() { if (event != nullptr) stream->record(event); }
        void synchronize() { if (event != nullptr) event->synchronize(); }

        void add_dependency(Instruction *inst) {
            Event* other_event = inst->event;
            if (other_event == nullptr) {
                inst->add_event();
                other_event = inst->event;
            }
            this->dependencies.push_back(other_event);
        }

        Layer* const to_layer;

    protected:
        Stream *stream;
        Event* event;
        std::vector<Event*> dependencies;
        int blocks, threads;
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
                  dst(state->get_second_order_input(second_order_node)),
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

        // Clear second order buffers
        ClearInstruction(DendriticNode *second_order_node,
            State *state, Stream *stream)
                : InitializeInstruction(second_order_node, state, stream) { }

        void activate() {
            Instruction::wait_for_dependencies();
            get_set_data().run(stream,
                blocks, threads,
                0.0, dst, size);
            Instruction::record_event();
        }
};

/* Adds noise to the input */
class NoiseInstruction : public InitializeInstruction {
    public:
        NoiseInstruction(Layer *layer, State *state, Stream *stream)
                : InitializeInstruction(layer, state, stream),
                  init(not layer->is_input()) { }

        void activate() {
            Instruction::wait_for_dependencies();
            get_randomize_data().run(stream,
                blocks, threads,
                dst, size,
                to_layer->noise_mean, to_layer->noise_std_dev,
                init);
            Instruction::record_event();
        }

    protected:
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
                  inter_device(state->is_inter_device(conn)),
                  d_to_d_event(nullptr),
                  inter_device_stream(nullptr),
                  activator(state->get_activator(conn, parent_node)) {
            if (inter_device) {
                src = state->get_output(conn->from_layer,
                        get_word_index(conn->delay,
                        state->get_output_type(conn->from_layer)));
                dst = state->get_device_output_buffer(conn);

                DeviceID source_device = state->get_device_id(conn->from_layer);
                inter_device_stream = ResourceManager::get_instance()
                    ->get_inter_device_stream(source_device);
                d_to_d_event = ResourceManager::get_instance()
                    ->create_event(source_device);
            }
        }

        void activate() {
            Instruction::wait_for_dependencies();
            if (inter_device) {
                src.copy_to(dst, inter_device_stream);
                inter_device_stream->record(d_to_d_event);
                stream->wait(d_to_d_event);
            }
            activator.run(stream,
                blocks, threads,
                synapse_data);
            Instruction::record_event();
        }

    protected:
        Kernel<SYNAPSE_ARGS> activator;
        Pointer<Output> src, dst;
        bool inter_device;
        Event *d_to_d_event;
        Stream *inter_device_stream;
};

/* Updates synaptic connection */
class SynapseUpdateInstruction : public SynapseInstruction {
    public:
        SynapseUpdateInstruction(DendriticNode *parent_node,
            Connection *conn, State *state, Stream *stream)
                : SynapseInstruction(parent_node, conn, state, stream),
                  updater(state->get_updater(conn, parent_node)) {
            if (conn->convolutional) {
                int num_weights = connection->get_num_weights();
                this->threads = calc_threads(num_weights);
                this->blocks = calc_blocks(num_weights);
            } else {
                this->threads = calc_threads(to_layer->size);
                this->blocks = calc_blocks(to_layer->size);
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

/* Computes second order dendritic node connection */
class SecondOrderDendriticInstruction : public Instruction {
    public:
        SecondOrderDendriticInstruction(DendriticNode *parent,
            State *state, Stream *stream)
                : Instruction(parent->to_layer, stream),
                  from_size(parent->get_second_order_size() / to_layer->size),
                  src(state->get_second_order_input(parent)),
                  dst(state->get_input(to_layer, parent->register_index)) { }

        void activate() {
            Instruction::wait_for_dependencies();
            get_calc_internal_second_order().run(
                stream, blocks, threads,
                from_size, to_layer->size, src, dst);
            Instruction::record_event();
        }

    protected:
        Pointer<float> src, dst;
        int from_size;
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

/* Transfers input data */
/* Sets input from buffer */
class InputTransferInstruction : public BufferedTransferInstruction<float> {
    public:
        InputTransferInstruction(Layer *layer, State *state,
            Environment *environment, Stream *stream)
                : BufferedTransferInstruction(layer, stream,
                      environment->buffer->get_input(layer),
                      state->get_buffer_input(layer),
                      state->get_input(layer),
                      environment->buffer) { }
};

/* Transfers expected data */
class ExpectedTransferInstruction : public BufferedTransferInstruction<Output> {
    public:
        ExpectedTransferInstruction(Layer *layer, State *state,
            Environment *environment, Stream *stream)
                : BufferedTransferInstruction(layer, stream,
                      environment->buffer->get_expected(layer),
                      state->get_buffer_expected(layer),
                      state->get_expected(layer),
                      environment->buffer) { }
};

/* Transfers output data */
class OutputTransferInstruction : public TransferInstruction<Output> {
    public:
        OutputTransferInstruction(Layer *layer, State *state,
            Environment *environment, Stream *stream)
                : TransferInstruction(layer, stream,
                      state->get_output(layer),
                      environment->buffer->get_output(layer)) { }
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

#endif
