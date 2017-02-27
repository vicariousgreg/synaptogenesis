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
        Instruction(Layer *layer);
        virtual ~Instruction();

        virtual void activate() = 0;
        virtual void update() { }
        virtual bool is_plastic() const { return false; }

        void record_events();

        Layer* const to_layer;

        void set_stream(Stream *stream) {
            delete this->stream;
            this->stream = stream;
        }
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
        InitializeInstruction(Layer *layer, State *state);

    protected:
        Pointer<float> dst;
};

/* Clears inputs */
class ClearInstruction : public InitializeInstruction {
    public:
        ClearInstruction(Layer *layer, State *state)
                : InitializeInstruction(layer, state) { }

        void activate();
};

/* Adds noise to the input */
class NoiseInstruction : public InitializeInstruction {
    public:
        NoiseInstruction(Layer *layer, State *state)
                : InitializeInstruction(layer, state),
                  init(layer->get_input_module() == NULL) { }

        void activate();

    protected:
        bool init;
};

/* Computes synaptic connection */
class SynapseInstruction : public Instruction {
    public:
        SynapseInstruction(Connection *conn, State *state);

        void activate();
        void update();
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
            DendriticNode *child, State *state);

        void activate();

    protected:
        Pointer<float> src, dst;
        bool init;
};

/* Transfers input data */
class InputTransferInstruction : public Instruction {
    public:
        InputTransferInstruction(Layer *to_layer, State *state,
            Environment *environment);

        void activate();

    protected:
        Pointer<float> src, dst;
};

/* Transfers output data */
class OutputTransferInstruction : public Instruction {
    public:
        OutputTransferInstruction(Layer *to_layer, State *state,
            Environment *environment);

        void activate();

    protected:
        Pointer<Output> src, dst;
};

/* Updates layer state */
class StateUpdateInstruction : public Instruction {
    public:
        StateUpdateInstruction(Layer *to_layer, State *state)
            : Instruction(to_layer),
              attribute_data(to_layer, state),
              attribute_kernel(state->get_attribute_kernel(to_layer)) { }

        void activate();

    protected:
        ATTRIBUTE_KERNEL attribute_kernel;
        const AttributeData attribute_data;
};

typedef std::vector<Instruction*> InstructionList;

#endif