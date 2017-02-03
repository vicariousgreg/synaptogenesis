#ifndef instruction_h
#define instruction_h

#include <vector>

#include "model/connection.h"
#include "state/state.h"
#include "engine/kernel/kernel_data.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/activator_kernel.h"
#include "engine/kernel/updater_kernel.h"
#include "util/parallel.h"

class Instruction {
    public:
        virtual bool is_plastic() const = 0;
        virtual void activate() = 0;
        virtual void update() = 0;

#ifdef PARALLEL
        void set_stream(cudaStream_t *stream) { this->stream = stream; }
        void add_event(cudaEvent_t *event) { this->events.push_back(event); }

    protected:
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        cudaStream_t *stream;
        std::vector<cudaEvent_t* > events;
#endif
};

class SynapseInstruction : public Instruction {
    public:
        SynapseInstruction(Connection *conn, State *state);

        void activate();
        void update();

        /* Learning related functions.
         * The connection's learning flag takes precedence for plasticity.
         * An instruction with a non-plastic connection cannot be made plastic
         *     for data allocation related reasons. */
        bool is_plastic() const { return kernel_data.plastic; }
        void enable_learning();

        const ConnectionType type;
        Connection* const connection;

    private:
        EXTRACTOR extractor;
        KERNEL activator;
        KERNEL updater;
        const KernelData kernel_data;
};

class DendriticInstruction : public Instruction {
    public:
        DendriticInstruction(DendriticNode *parent,
            DendriticNode *child, State *state);

        bool is_plastic() const { return false; }
        void activate();
        void update() { }

        Layer* const to_layer;
        const int size;
        const bool clear;

    private:
        float *src, *dst;
};

typedef std::vector<Instruction*> InstructionList;

#endif
