#include "io/environment.h"
#include "io/buffer.h"
#include "io/module/module.h"
#include "io/module/visualizer_input_module.h"
#include "io/module/visualizer_output_module.h"
#include "model/model.h"
#include "state/state.h"
#include "visualizer.h"

Environment::Environment(State *state)
        : state(state), visualizer(nullptr),
          buffer(build_buffer(
              ResourceManager::get_instance()->get_host_id(), state->model)) {
    // Extract modules
    for (auto& layer : state->model->get_layers()) {
        bool visualizer_input = false;
        bool visualizer_output = false;

        // Add input module
        // If visualizer input module, set flag
        Module *input_module = layer->get_input_module();
        if (input_module != nullptr) {
            this->input_modules.push_back(input_module);
            visualizer_input =
                dynamic_cast<VisualizerInputModule*>(input_module) != nullptr;
        }

        // Add expected module
        Module *expected_module = layer->get_expected_module();
        if (expected_module != nullptr)
            this->expected_modules.push_back(expected_module);

        // Add output modules
        // If visualizer output module is found, set flag
        auto output_modules = layer->get_output_modules();
        for (auto& output_module : output_modules) {
            this->output_modules.push_back(output_module);
            visualizer_output |=
                dynamic_cast<VisualizerOutputModule*>(output_module) != nullptr;
        }

        if (visualizer_input or visualizer_output) {
            if (visualizer == nullptr) visualizer = new Visualizer(this);
            visualizer->add_layer(layer, visualizer_input, visualizer_output);
        }
    }
}

Environment::~Environment() {
    delete buffer;
    for (auto& module : this->input_modules) delete module;
    for (auto& module : this->output_modules) delete module;
    if (visualizer != nullptr) delete visualizer;
}

OutputType Environment::get_output_type(Layer *layer) {
    return state->get_output_type(layer);
}

void Environment::step_input() {
    for (auto& module : this->input_modules)
        module->feed_input(buffer);

    for (auto& module : this->expected_modules)
        module->feed_expected(buffer);
}

void Environment::step_output() {
    for (auto& module : this->output_modules)
        module->report_output(buffer, state->get_output_type(module->layer));
}

void Environment::ui_launch() {
    if (visualizer != nullptr) visualizer->launch();
}

void Environment::ui_update() {
    if (visualizer != nullptr) visualizer->update();
}
