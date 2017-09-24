#include "engine/report.h"
#include "engine/engine.h"
#include "io/module.h"
#include "state/state.h"
#include "network/layer.h"
#include "network/structure.h"

Report::Report(Engine* engine, State* state, int iterations, float total_time)
    : iterations(iterations),
      total_time(total_time),
      average_time(total_time / iterations),
      network_bytes(state->get_network_bytes()),
      state_buffer_bytes(state->get_buffer_bytes()),
      engine_buffer_bytes(engine->get_buffer_bytes()) { }

Report::~Report() {
    for (auto reps : layer_reports)
        for (auto r : reps.second)
            delete r;
}

void Report::print() {
    printf("\n\n* Engine Report:\n\n");
    printf("Total time: %fs\n", total_time);
    printf("Time averaged over %d iterations: %fs\n",
           iterations, average_time);
    printf("Network state size: %12zu bytes    (%12f MB)\n",
        network_bytes, (float)network_bytes / (1024 * 1024));
    printf("State  buffer size: %12zu bytes    (%12f MB)\n",
        state_buffer_bytes, (float)state_buffer_bytes / (1024 * 1024));
    printf("Engine buffer size: %12zu bytes    (%12f MB)\n",
        engine_buffer_bytes, (float)engine_buffer_bytes / (1024 * 1024));

    for (auto layer_pair : layer_reports) {
        printf("\nReport for %s\n", layer_pair.first->str().c_str());
        for (auto rep : layer_pair.second) {
            printf("  Module: %s\n", rep->module.c_str());
            for (auto pair : rep->properties->get())
                printf("  %15s -> %s\n",
                    pair.first.c_str(),
                    pair.second.c_str());
        }
    }
}

void Report::add_report(Module *module, Layer *layer, PropertyConfig *props) {
    this->layer_reports[layer].push_back(
        new LayerReport(
            module->get_name(),
            layer->structure->name,
            layer->name,
            props));
}
