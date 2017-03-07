#ifndef instruction_h
#define instruction_h

#include <vector>

#include "model/connection.h"
#include "state/state.h"
#include "state/attributes.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/synapse_kernel.h"
#include "engine/kernel/activator_kernel.h"
#include "util/parallel.h"

class Instruction {
    public:
        Instruction(Layer *layer)
                : to_layer(layer),
                  activator_threads(calc_threads(layer->size)),
                  activator_blocks(calc_blocks(layer->size)),
                  stream(Stream::get_default_stream()) { }

        virtual void activate() = 0;
        virtual void update() { }
        virtual bool is_plastic() const { return false; }

        void record_events() {
            for (auto& event : events) stream->record(event);
        }

        Layer* const to_layer;

        void set_stream(Stream *stream) { this->stream = stream; }
        void add_event(Event *event) { this->events.push_back(event); }

    protected:
        Stream *stream;
        std::vector<Event*> events;
        int activator_blocks, activator_threads;
        int updater_blocks, updater_threads;
};

/* Instructions that initialize the input without connections */
class InitializeInstruction : public Instruction {
    public:
        InitializeInstruction(Layer *layer, State *state)
                : Instruction(layer),
                  dst(state->get_input(layer)) { }

    protected:
        Pointer<float> dst;
};

/* Clears inputs */
class ClearInstruction : public InitializeInstruction {
    public:
        ClearInstruction(Layer *layer, State *state)
                : InitializeInstruction(layer, state) { }

        void activate() {
            stream->run_kernel(clear_data,
                activator_blocks, activator_threads,
                dst, to_layer->size);
            Instruction::record_events();
        }
};

/* Adds noise to the input */
class NoiseInstruction : public InitializeInstruction {
    public:
        NoiseInstruction(Layer *layer, State *state)
                : InitializeInstruction(layer, state),
                  init(layer->get_input_module() == nullptr) { }

        void activate() {
            stream->run_kernel(randomize_data,
                activator_blocks, activator_threads,
                dst, to_layer->size, to_layer->noise, init);
            Instruction::record_events();
        }

    protected:
        bool init;
};

/* Computes synaptic connection */
class SynapseInstruction : public Instruction {
    public:
        SynapseInstruction(Connection *conn, State *state)
                : Instruction(conn->to_layer),
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
            stream->run_kernel(activator,
                activator_blocks, activator_threads,
                synapse_data);
            Instruction::record_events();
        }

        void update() {
            if (this->updater != nullptr)
                stream->run_kernel(updater,
                    updater_blocks, updater_threads,
                    synapse_data);
        }

        bool is_plastic() const { return synapse_data.plastic; }

        const ConnectionType type;
        Connection* const connection;

    protected:
        EXTRACTOR extractor;
        SYNAPSE_KERNEL activator;
        SYNAPSE_KERNEL updater;
        const SynapseData synapse_data;
};

/* Computes dendritic node connection */
class DendriticInstruction : public Instruction {
    public:
        DendriticInstruction(DendriticNode *parent,
            DendriticNode *child, State *state)
                : Instruction(parent->to_layer),
                  init(child->register_index != 0),
                  src(state->get_input(to_layer, child->register_index)),
                  dst(state->get_input(to_layer, parent->register_index)) { }

        void activate() {
            stream->run_kernel(calc_internal,
                activator_blocks, activator_threads,
                to_layer->size, src, dst, init);
            Instruction::record_events();
        }

    protected:
        Pointer<float> src, dst;
        bool init;
};

/* Transfers data */
template<class T>
class TransferInstruction : public Instruction {
    public:
        TransferInstruction(Layer *layer, Pointer<T> src, Pointer<T> dst)
                : Instruction(layer),
                  src(src), dst(dst) { }

        virtual void activate() {
            this->stream->transfer(src, dst);
            Instruction::record_events();
        }

    protected:
        Pointer<T> src, dst;
};

/* Transfers input data */
class InputTransferInstruction : public TransferInstruction<float> {
    public:
        InputTransferInstruction(Layer *layer, State *state,
            Environment *environment)
                : TransferInstruction(layer,
                      environment->buffer->get_input(layer),
                      state->buffer->get_input(layer)),
                  buffer(environment->buffer) { }

        virtual void activate() {
            // Only transfer if the buffer is dirty
            if (buffer->get_dirty(to_layer)) {
                buffer->set_dirty(to_layer, false);
                TransferInstruction<float>::activate();
            } else {
                Instruction::record_events();
            }
        }

    protected:
        Buffer* buffer;
};

/* Sets input from buffer */
class InternalInputTransferInstruction : public TransferInstruction<float> {
    public:
        InternalInputTransferInstruction(Layer *layer, State *state)
                : TransferInstruction(layer,
                      state->buffer->get_input(layer),
                      state->get_input(layer)) { }
};

/* Transfers output data */
class OutputTransferInstruction : public TransferInstruction<Output> {
    public:
        OutputTransferInstruction(Layer *layer,
            State *state, Environment *environment)
                : TransferInstruction(layer,
                      state->buffer->get_output(layer),
                      environment->buffer->get_output(layer)) { }
};

/* Sets output to buffer */
class InternalOutputTransferInstruction : public TransferInstruction<Output> {
    public:
        InternalOutputTransferInstruction(Layer *layer, State *state)
                : TransferInstruction(layer,
                      state->get_output(layer),
                      state->buffer->get_output(layer)) { }
};

/* Updates layer state */
class StateUpdateInstruction : public Instruction {
    public:
        StateUpdateInstruction(Layer *to_layer, State *state)
            : Instruction(to_layer),
              attribute_data(to_layer, state),
              attribute_kernel(state->get_attribute_kernel(to_layer)) { }

        void activate() {
            stream->run_kernel(attribute_kernel,
                activator_blocks, activator_threads,
                attribute_data);
            Instruction::record_events();
        }

    protected:
        ATTRIBUTE_KERNEL attribute_kernel;
        const AttributeData attribute_data;
};

typedef std::vector<Instruction*> InstructionList;

#endif
