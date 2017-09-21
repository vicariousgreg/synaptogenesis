#include "engine/report.h"
#include "engine/engine.h"
#include "state/state.h"

Report::Report(Engine* engine, State* state, int iterations, float total_time)
    : iterations(iterations),
      total_time(total_time),
      average_time(total_time / iterations),
      network_bytes(state->get_network_bytes()),
      state_buffer_bytes(state->get_buffer_bytes()),
      engine_buffer_bytes(engine->get_buffer_bytes()) { }

void Report::print() {
    printf("\n\n* Engine Report:\n\n");
    printf("Total time: %f\n", total_time);
    printf("Time averaged over %d iterations: %f\n",
           iterations, average_time);
    printf("Network state size: %12zu bytes    (%12f MB)\n",
        network_bytes, (float)network_bytes / (1024 * 1024));
    printf("State  buffer size: %12zu bytes    (%12f MB)\n",
        state_buffer_bytes, (float)state_buffer_bytes / (1024 * 1024));
    printf("Engine buffer size: %12zu bytes    (%12f MB)\n",
        engine_buffer_bytes, (float)engine_buffer_bytes / (1024 * 1024));
}
