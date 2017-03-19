#ifndef instruction_h
#define instruction_h

#include <vector>

#include "model/connection.h"
#include "state/state.h"
#include "state/attributes.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/attribute_data.h"
#include "engine/kernel/synapse_kernel.h"

class Instruction {
    public:
        Instruction(Layer *layer, Stream *stream)
                : to_layer(layer),
                  activator_threads(calc_threads(layer->size)),
                  activator_blocks(calc_blocks(layer->size)),
                  stream(stream),
                  event(nullptr) { }
        virtual ~Instruction() { }

        virtual void activate() = 0;
        virtual void update() { }
        virtual bool is_plastic() const { return false; }

        void wait_for_dependencies() {
            for (auto& dep : dependencies) stream->wait(dep);
        }

        void record_event() { if (event != nullptr) stream->record(event); }
        void synchronize() { if (event != nullptr) event->synchronize(); }

        void add_dependency(Instruction *inst) {
            Event* other_event = inst->event;
            if (other_event == nullptr) {
                other_event = ResourceManager::get_instance()->create_event(
                    inst->stream->get_device_id());
                inst->event = other_event;
            }
            this->dependencies.push_back(other_event);
        }

        Layer* const to_layer;

    protected:
        Stream *stream;
        Event* event;
        std::vector<Event*> dependencies;
        int activator_blocks, activator_threads;
        int updater_blocks, updater_threads;
};

/* Instructions that initialize the input without connections */
class InitializeInstruction : public Instruction {
    public:
        InitializeInstruction(Layer *layer, State *state, Stream *stream)
                : Instruction(layer, stream),
                  dst(state->get_input(layer)) { }

    protected:
        Pointer<float> dst;
};

/* Clears inputs */
class ClearInstruction : public InitializeInstruction {
    public:
        ClearInstruction(Layer *layer, State *state, Stream *stream)
                : InitializeInstruction(layer, state, stream) { }

        void activate() {
            Instruction::wait_for_dependencies();
            get_clear_data().run(stream,
                activator_blocks, activator_threads,
                dst, to_layer->size);
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
                activator_blocks, activator_threads,
                dst, to_layer->size, to_layer->noise, init);
            Instruction::record_event();
        }

    protected:
        bool init;
};

/* Computes synaptic connection */
class SynapseInstruction : public Instruction {
    public:
        SynapseInstruction(Connection *conn, State *state, Stream *stream)
                : Instruction(conn->to_layer, stream),
                  connection(conn),
                  synapse_data(conn, state),
                  type(conn->type),
                  activator(state->get_activator(conn)),
                  updater((conn->plastic) ? state->get_updater(conn) : nullptr) {
            if (conn->convolutional) {
                int num_weights = connection->get_num_weights();
                this->updater_threads = calc_threads(num_weights);
                this->updater_blocks = calc_blocks(num_weights);
            } else {
                this->updater_threads = calc_threads(to_layer->size);
                this->updater_blocks = calc_blocks(to_layer->size);
            }
        }

        void activate() {
            Instruction::wait_for_dependencies();
            activator.run(stream,
                activator_blocks, activator_threads,
                synapse_data);
            Instruction::record_event();
        }

        void update() {
            if (this->is_plastic())
                updater.run(stream,
                    updater_blocks, updater_threads,
                    synapse_data);
        }

        bool is_plastic() const { return synapse_data.plastic; }

        const ConnectionType type;
        Connection* const connection;

    protected:
        EXTRACTOR extractor;
        Kernel<SYNAPSE_ARGS>activator;
        Kernel<SYNAPSE_ARGS>updater;
        SynapseData synapse_data;
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
            get_calc_internal().run(stream,
                activator_blocks, activator_threads,
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
        TransferInstruction(Layer *layer,
            Pointer<T> src, Pointer<T> dst, Stream *stream)
                : Instruction(layer, stream),
                  src(src), dst(dst) { }

        virtual void activate() {
            Instruction::wait_for_dependencies();
            src.copy_to(dst, stream);
            Instruction::record_event();
        }

    protected:
        Pointer<T> src, dst;
};

/* Transfers input data */
class InputTransferInstruction : public TransferInstruction<float> {
    public:
        InputTransferInstruction(Layer *layer, State *state,
            Environment *environment, Stream *stream)
                : TransferInstruction(layer,
                      environment->buffer->get_input(layer),
                      state->get_buffer_input(layer),
                      stream),
                  buffer(environment->buffer) { }

        virtual void activate() {
            // Only transfer if the buffer is dirty
            if (buffer->get_dirty(to_layer)) {
                buffer->set_dirty(to_layer, false);
                TransferInstruction<float>::activate();
            } else {
                Instruction::wait_for_dependencies();
                Instruction::record_event();
            }
        }

    protected:
        Buffer* buffer;
};

/* Sets input from buffer */
class InternalInputTransferInstruction : public TransferInstruction<float> {
    public:
        InternalInputTransferInstruction(Layer *layer,
            State *state, Stream *stream)
                : TransferInstruction(layer,
                      state->get_buffer_input(layer),
                      state->get_input(layer),
                      stream) { }
};

/* Transfers expected data */
class ExpectedTransferInstruction : public TransferInstruction<Output> {
    public:
        ExpectedTransferInstruction(Layer *layer, State *state,
            Environment *environment, Stream *stream)
                : TransferInstruction(layer,
                      environment->buffer->get_expected(layer),
                      state->get_buffer_expected(layer),
                      stream),
                  buffer(environment->buffer) { }

        virtual void activate() {
            // Only transfer if the buffer is dirty
            if (buffer->get_dirty(to_layer)) {
                buffer->set_dirty(to_layer, false);
                TransferInstruction<Output>::activate();
            } else {
                Instruction::wait_for_dependencies();
                Instruction::record_event();
            }
        }

    protected:
        Buffer* buffer;
};

/* Sets expected from buffer */
class InternalExpectedTransferInstruction : public TransferInstruction<Output> {
    public:
        InternalExpectedTransferInstruction(Layer *layer,
            State *state, Stream *stream)
                : TransferInstruction(layer,
                      state->get_buffer_expected(layer),
                      state->get_expected(layer),
                      stream) { }
};

/* Transfers output data */
class OutputTransferInstruction : public TransferInstruction<Output> {
    public:
        OutputTransferInstruction(Layer *layer,
            State *state, Environment *environment, Stream *stream)
                : TransferInstruction(layer,
                      state->get_buffer_output(layer),
                      environment->buffer->get_output(layer),
                      stream) { }
};

/* Sets output to buffer */
class InternalOutputTransferInstruction : public TransferInstruction<Output> {
    public:
        InternalOutputTransferInstruction(Layer *layer,
            State *state, Stream *stream)
                : TransferInstruction(layer,
                      state->get_output(layer),
                      state->get_buffer_output(layer),
                      stream) { }
};

/* Transfers outputs between devices */
class DeviceToDeviceTransferFunction : public TransferInstruction<Output> {
    public:
        DeviceToDeviceTransferFunction(Connection *conn,
            State *state, Stream *stream)
                : TransferInstruction(conn->to_layer,
                  state->get_output(conn->from_layer,
                      get_word_index(conn->delay,
                          state->get_output_type(conn->from_layer))),
                  state->get_device_output_buffer(conn),
                  stream) { }
};

/* Updates layer state */
class StateUpdateInstruction : public Instruction {
    public:
        StateUpdateInstruction(Layer *to_layer, State *state, Stream *stream)
            : Instruction(to_layer, stream),
              attribute_data(to_layer, state),
              attribute_kernel(state->get_attribute_kernel(to_layer)) { }

        void activate() {
            Instruction::wait_for_dependencies();
            attribute_kernel.run(stream,
                activator_blocks, activator_threads,
                attribute_data);
            Instruction::record_event();
        }

    protected:
        Kernel<ATTRIBUTE_ARGS> attribute_kernel;
        AttributeData attribute_data;
};

typedef std::vector<Instruction*> InstructionList;

#endif
