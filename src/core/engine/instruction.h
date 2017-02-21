#ifndef instruction_h
#define instruction_h

#include <vector>

#include "model/connection.h"
#include "state/state.h"
#include "state/attributes.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/activator_kernel.h"
#include "util/parallel.h"

class Instruction {
    public:
        Instruction(Layer *layer);

        virtual void activate() = 0;
        virtual void update() { }
        virtual bool is_plastic() const { return false; }

        Layer* const to_layer;

#ifdef PARALLEL
        void set_stream(cudaStream_t stream) { this->stream = stream; }
        void add_event(cudaEvent_t event) { this->events.push_back(event); }
        void record_events();

    protected:
        dim3 activator_blocks, activator_threads;
        dim3 updater_blocks, updater_threads;
        cudaStream_t stream;
        std::vector<cudaEvent_t> events;
#endif
};

/* Instructions that initialize the input without connections */
class InitializeInstruction : public Instruction {
    public:
        InitializeInstruction(Layer *layer, State *state);

    protected:
        float *dst;
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
        KERNEL activator;
        KERNEL updater;
        const SynapseData synapse_data;
};

/* Computes dendritic node connection */
class DendriticInstruction : public Instruction {
    public:
        DendriticInstruction(DendriticNode *parent,
            DendriticNode *child, State *state);

        void activate();

    protected:
        float *src, *dst;
        bool init;
};

/* Transfers input data */
class InputTransferInstruction : public Instruction {
    public:
        InputTransferInstruction(Layer *to_layer, State *state);

        void activate();

    protected:
        float *src, *dst;
};

/* Transfers output data */
class OutputTransferInstruction : public Instruction {
    public:
        OutputTransferInstruction(Layer *to_layer, State *state);

        void activate();

    protected:
        Output *src, *dst;
};

/* Updates layer state */
class StateUpdateInstruction : public Instruction {
    public:
        StateUpdateInstruction(Layer *to_layer, State *state)
            : Instruction(to_layer),
              start_index(state->get_start_index(to_layer)),
              attributes(state->get_attributes_pointer()),
              attribute_kernel(state->get_attribute_kernel()) { }

        void activate();

    protected:
        const int start_index;
        const Attributes *attributes;
        ATTRIBUTE_KERNEL attribute_kernel;
};

typedef std::vector<Instruction*> InstructionList;

#endif
