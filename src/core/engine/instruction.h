#ifndef instruction_h
#define instruction_h

#include <vector>

#include "network/connection.h"
#include "state/state.h"
#include "state/attributes.h"
#include "io/buffer.h"
#include "engine/engine.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/attribute_data.h"
#include "util/resources/scheduler.h"
#include "util/transpose.h"

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

        void activate() {
            Instruction::wait_for_dependencies();
            this->activate_impl();
            Instruction::record_event();
        }
        virtual void activate_impl() = 0;

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
        void synchronize() {
            if (event != nullptr)
                Scheduler::get_instance()->synchronize(event);
        }

        void add_dependency(Instruction *inst) {
            // Only add dependencies for instructions on different streams
            if (inst->stream != this->stream) {
                inst->add_event();
                this->dependencies.push_back(inst->event);
            }
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
        dim3 blocks, threads;
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

        void activate_impl() {
            get_copy_pointer_kernel<Output>().schedule(
                stream, 0, 0, src, dst, stream);
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
        InitializeInstruction(Layer *layer, State *state, Stream *stream,
            bool overwrite)
                : Instruction(layer, stream),
                  dst(state->get_input(layer)),
                  size(layer->size),
                  overwrite(overwrite) { }

        // Initialize layer register (internal dendritic node)
        InitializeInstruction(DendriticNode *node, State *state, Stream *stream)
                : Instruction(node->to_layer, stream),
                  dst(state->get_input(node->to_layer, node->register_index)),
                  size(node->to_layer->size),
                  overwrite(true) { }

    protected:
        Pointer<float> dst;
        int size;
        bool overwrite;
};

/* Instructions that initialize other attribute data without connections */
class AuxiliaryInitializeInstruction : public Instruction {
    public:
        // Initialize layer buffers
        AuxiliaryInitializeInstruction(Layer *layer, State *state,
            Stream *stream, std::string key, float val, bool overwrite)
                : Instruction(layer, stream),
                  dst(state->get_neuron_data(layer, key)),
                  size(layer->size),
                  val(val),
                  overwrite(overwrite) { }

        void activate_impl() {
            get_set_data().schedule(stream,
                blocks, threads,
                val, dst, size, overwrite);
        }

    protected:
        Pointer<float> dst;
        int size;
        float val;
        bool overwrite;
};

/* Clears inputs */
class SetInstruction : public InitializeInstruction {
    public:
        SetInstruction(Layer *layer, State *state, Stream *stream,
            float val, bool overwrite)
                : InitializeInstruction(layer, state, stream, overwrite),
                  val(val) { }

        // Constructor for dendritic node register setting
        SetInstruction(DendriticNode *node, State *state, Stream *stream,
            float val)
                : InitializeInstruction(node, state, stream),
                  val(val) { }

        // Constructor for flat init config
        SetInstruction(Layer *layer, State *state, Stream *stream,
            bool overwrite)
                : InitializeInstruction(layer, state, stream, overwrite) {
            auto config = layer->get_config()->get_child("init config");
            val = config->get_float("value", 1.0);
        }

        void activate_impl() {
            get_set_data().schedule(stream,
                blocks, threads,
                val, dst, size, overwrite);
        }

    protected:
        float val;
};

/* Adds noise to the input */
class UniformNoiseInstruction : public InitializeInstruction {
    public:
        UniformNoiseInstruction(Layer *layer, State *state,
            Stream *stream, bool overwrite)
                : InitializeInstruction(layer, state, stream, overwrite) {
            auto config = layer->get_config()->get_child("init config");
            min = config->get_float("min", 0.0);
            max = config->get_float("max", 1.0);
        }

        void activate_impl() {
            get_randomize_data_uniform().schedule(stream,
                blocks, threads,
                dst, size,
                min, max,
                overwrite);
        }

    protected:
        float min, max;
};

class NormalNoiseInstruction : public InitializeInstruction {
    public:
        NormalNoiseInstruction(Layer *layer, State *state,
            Stream *stream, bool overwrite)
                : InitializeInstruction(layer, state, stream, overwrite) {
            auto config = layer->get_config()->get_child("init config");
            mean = config->get_float("mean", 1.0);
            std_dev = config->get_float("std dev", 0.1);
        }

        void activate_impl() {
            get_randomize_data_normal().schedule(stream,
                blocks, threads,
                dst, size,
                mean, std_dev,
                overwrite);
        }

    protected:
        float mean;
        float std_dev;
};

class PoissonNoiseInstruction : public InitializeInstruction {
    public:
        PoissonNoiseInstruction(Layer *layer, State *state,
            Stream *stream, bool overwrite)
                : InitializeInstruction(layer, state, stream, overwrite) {
            auto config = layer->get_config()->get_child("init config");
            val = config->get_float("value", 20);
            rate = 0.001 * config->get_float("rate", 1);

            if (config->get_bool("random", "false")) {
                random_rates = Pointer<float>(size);
                fRand(random_rates.get(), size, 0.0, rate);
                random_rates.transfer(stream->get_device_id(), nullptr);
            }
        }

        void activate_impl() {
            get_randomize_data_poisson().schedule(stream,
                blocks, threads,
                dst, size,
                val, rate,
                overwrite, random_rates);
        }

    protected:
        float val;
        float rate;
        Pointer<float> random_rates;
};

/* Operates on synapses */
class SynapseInstruction : public Instruction {
    public:
        SynapseInstruction(DendriticNode* parent_node, Connection *conn,
            State *state, Stream *stream, bool updater)
                : Instruction(conn->to_layer, stream),
                  connection(conn),
                  synapse_data(parent_node, conn, state, updater) { }

        Connection* const connection;

    protected:
        SynapseData synapse_data;
};

/* Activates synaptic connection */
class SynapseActivateInstruction : public SynapseInstruction {
    public:
        SynapseActivateInstruction(DendriticNode *parent_node,
            Connection *conn, State *state, Stream *stream)
                : SynapseInstruction(parent_node, conn, state, stream, false),
                  activators(state->get_activators(conn)) {
            // Convolutional activate instructions iterate over weights
            // This is because of special conditions (see connection.cpp)
            if (conn->second_order_slave and conn->convolutional) {
                int num_weights = connection->get_num_weights();
                this->threads = calc_threads(num_weights);
                this->blocks = calc_blocks(num_weights);
            } else if (conn->get_type() == SUBSET) {
                int num_dest = conn->config->get_subset_config().total_size;
                this->threads = calc_threads(num_dest);
                this->blocks = calc_blocks(num_dest);
            }
        }

        void activate_impl() {
            for (auto& activator : activators)
                activator.schedule(stream,
                    blocks, threads,
                    synapse_data);
        }

    protected:
        KernelList<SYNAPSE_ARGS> activators;
};

/* Updates synaptic connection */
class SynapseUpdateInstruction : public SynapseInstruction {
    public:
        SynapseUpdateInstruction(DendriticNode *parent_node,
            Connection *conn, State *state, Stream *stream)
                : SynapseInstruction(parent_node, conn, state, stream, true),
                  updaters(state->get_updaters(conn)) {
            // Convolutional update instructions iterate over weights
            if (conn->convolutional) {
                int num_weights = connection->get_num_weights();
                this->threads = calc_threads(num_weights);
                this->blocks = calc_blocks(num_weights);
            } else if (conn->get_type() == SUBSET) {
                int num_dest = conn->config->get_subset_config().total_size;
                this->threads = calc_threads(num_dest);
                this->blocks = calc_blocks(num_dest);
            }
        }

        void activate_impl() {
            for (auto& updater : updaters)
                updater.schedule(stream,
                    blocks, threads,
                    synapse_data);
        }

    protected:
        KernelList<SYNAPSE_ARGS> updaters;
};

/* Computes dendritic node connection */
class DendriticInstruction : public Instruction {
    public:
        DendriticInstruction(DendriticNode *parent,
            DendriticNode *child, State *state, Stream *stream)
                : Instruction(parent->to_layer, stream),
                  aggregator(get_aggregator(child->opcode, stream->get_device_id())),
                  src(state->get_input(to_layer, child->register_index)),
                  dst(state->get_input(to_layer, parent->register_index)),
                  trail_value(child->init_val) { }

        void activate_impl() {
            get_calc_internal().schedule(
                stream, blocks, threads,
                to_layer->size, src, dst, aggregator, trail_value);
        }

    protected:
        Pointer<float> src, dst;
        AGGREGATOR aggregator;
        float trail_value;
};

/* Transposes a matrix */
class TransposeInstruction : public Instruction {
    public:
        TransposeInstruction(Connection *conn, State *state, Stream *stream)
                : Instruction(conn->to_layer, stream),
                  connection(conn),
                  matrix(state->get_matrix(conn)) {
                this->threads = calc_transpose_threads(
                    matrix->get_rows(), matrix->get_columns());
                this->blocks = calc_transpose_blocks(
                    matrix->get_rows(), matrix->get_columns());
        }

        void activate_impl() {
            get_transposer().schedule(
                stream, blocks, threads,
                matrix->get_weights(), matrix->get_weights_transposed(),
                matrix->get_rows(), matrix->get_columns());
        }

        Connection* const connection;
        const WeightMatrix * const matrix;

    protected:
        Kernel<const Pointer<float>, Pointer<float>,
            const int, const int> transposer;
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

        void activate_impl() {
            get_copy_pointer_kernel<T>().schedule(
                stream, 0, 0, src, dst, stream);
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
            Buffer *source_buffer)
                : Instruction(layer, stream),
                  src(src), inter(inter), dst(dst),
                  source_buffer(source_buffer) {
            this->add_event();
        }

        virtual bool is_dirty() = 0;

        void activate_impl() {
            if (is_dirty())
                get_copy_pointer_kernel<T>().schedule(
                    stream, 0, 0, src, inter, stream);

            get_copy_pointer_kernel<T>().schedule(
                stream, 0, 0, inter, dst, stream);
        }

    protected:
        Pointer<T> src, inter, dst;
        Buffer *source_buffer;
};

/* Transfers weights intradevice for second order connections */
class SecondOrderWeightTransferInstruction : public TransferInstruction<float> {
    public:
        SecondOrderWeightTransferInstruction(DendriticNode *node,
            State *state, Stream *stream)
                : TransferInstruction(
                      node->get_second_order_connection()->to_layer,
                      stream,
                      state->get_weights(node->get_second_order_connection()),
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

        virtual bool is_dirty() {
            bool dirty = source_buffer->get_input_dirty(to_layer);
            if (dirty) source_buffer->set_input_dirty(to_layer, false);
            return dirty;
        }
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

/* Transfers auxiliary input data */
class InputAuxiliaryTransferInstruction : public TransferInstruction<float> {
    public:
        InputAuxiliaryTransferInstruction(std::string key,
            Layer *layer, State *state,
            Engine *engine, Stream *stream)
                : TransferInstruction(layer, stream,
                      Pointer<float>(engine->get_buffer()
                          ->get_input_auxiliary(layer, key)),
                      Pointer<float>(state->get_neuron_data(layer, key))) { }
};

/* Transfers auxiliary output data */
class OutputAuxiliaryTransferInstruction : public TransferInstruction<float> {
    public:
        OutputAuxiliaryTransferInstruction(std::string key,
            Layer *layer, State *state,
            Engine *engine, Stream *stream)
                : TransferInstruction(layer, stream,
                      Pointer<float>(state->get_neuron_data(layer, key)),
                      Pointer<float>(engine->get_buffer()
                          ->get_output_auxiliary(layer, key))) { }
};

/* Operates on neuron state */
class StateInstruction : public Instruction {
    public:
        StateInstruction(Layer *to_layer, State *state, Stream *stream,
            Kernel<ATTRIBUTE_ARGS> attribute_kernel)
            : Instruction(to_layer, stream),
              attribute_data(to_layer, state),
              attribute_kernel(attribute_kernel) { }

        void activate_impl() {
            attribute_kernel.schedule(stream,
                blocks, threads,
                attribute_data);
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

/* Retrieves the appropriate initialization instruction, based on init config
 *   specification and whether the layer has input from the environment */
InitializeInstruction* get_initialize_instruction(
        Layer *layer, State *state, Stream *stream, bool is_input);

typedef std::vector<Instruction*> InstructionList;
typedef std::vector<SynapseInstruction*> SynapseInstructionList;

#endif
