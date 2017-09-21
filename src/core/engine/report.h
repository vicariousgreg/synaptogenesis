#ifndef report_h
#define report_h

#include "state/state.h"
#include "engine/engine.h"

class Report {
    public:
        Report(Engine* engine, State* state, int iterations, float total_time)
            : iterations(iterations),
              total_time(total_time),
              average_time(total_time / iterations),
              network_bytes(state->get_network_bytes()),
              state_buffer_bytes(state->get_buffer_bytes()),
              engine_buffer_bytes(engine->get_buffer_bytes()) { }

        const int iterations;
        const float total_time;
        const float average_time;
        const size_t network_bytes;
        const size_t state_buffer_bytes;
        const size_t engine_buffer_bytes;
};

#endif
