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
        Instruction(Connection *conn, State *state);

        bool is_plastic() { return this->kernel_data.plastic; }
        void disable_learning();
        void activate();
        void update();

#ifdef PARALLEL
        void set_stream(cudaStream_t *stream) { this->stream = stream; }
        void add_event(cudaEvent_t *event) { this->events.push_back(event); }

        dim3 blocks_per_grid;
        dim3 threads_per_block;
        cudaStream_t *stream;
        std::vector<cudaEvent_t* > events;
#endif

        ConnectionType type;

        EXTRACTOR extractor;
        KERNEL activator;
        KERNEL updater;

        Connection *connection;
        KernelData kernel_data;
};

#endif
