#include "engine/report.h"
#include "engine/engine.h"
#include "io/module.h"
#include "state/state.h"
#include "network/layer.h"
#include "network/structure.h"

Report::Report(Engine* engine, State* state, size_t iterations, float total_time) {
    this->set("iterations", std::to_string(iterations));
    this->set("total time", total_time);
    this->set("average time", total_time / iterations);
    this->set("fps", iterations / total_time);
    this->set("network bytes", std::to_string(state->get_network_bytes()));
    this->set("state buffer bytes", std::to_string(state->get_buffer_bytes()));
    this->set("engine buffer bytes", std::to_string(engine->get_buffer_bytes()));
}

void Report::print() {
    const size_t iterations = std::stoll(get("iterations"));
    const float total_time = get_float("total time", 0.0);
    const float average_time = get_float("average time", 0.0);
    const float fps = get_float("fps", 0.0);
    const size_t network_bytes = std::stoll(get("network bytes"));
    const size_t state_buffer_bytes = std::stoll(get("state buffer bytes"));
    const size_t engine_buffer_bytes = std::stoll(get("engine buffer bytes"));

    printf("\n\n* Engine Report:\n\n");
    printf("Total time: %fs\n", total_time);
    printf("Time averaged over %d iterations: %fs (%6.2ffps)\n",
           iterations, average_time, fps);
    printf("Network state size: %12zu bytes    (%12f MB)\n",
        network_bytes, (float)network_bytes / (1024 * 1024));
    printf("State  buffer size: %12zu bytes    (%12f MB)\n",
        state_buffer_bytes, (float)state_buffer_bytes / (1024 * 1024));
    printf("Engine buffer size: %12zu bytes    (%12f MB)\n",
        engine_buffer_bytes, (float)engine_buffer_bytes / (1024 * 1024));

    auto arr = get_array("layer reports");
    for (auto indices : layer_indices) {
        printf("\nReport for %s\n", indices.first->str().c_str());

        for (auto index : indices.second) {
            for (auto pair : arr[index]->get())
                printf("  %15s -> %s\n",
                    pair.first.c_str(),
                    pair.second.c_str());
        }
    }
    printf("\n");
}

void Report::add_report(Module *module, Layer *layer, PropertyConfig props) {
    PropertyConfig layer_report;
    layer_report.set("Module", module->get_name());
    layer_report.set("Structure", layer->structure->name);
    layer_report.set("Layer", layer->name);
    for (auto pair : props.get())
        layer_report.set(pair.first, pair.second);

    // Add the index of the report
    layer_indices[layer].push_back(get_array("layer reports").size());

    // Add the report
    this->add_to_array("layer reports", &layer_report);
}
