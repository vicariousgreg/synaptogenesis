#include "io/environment.h"
#include "io/buffer.h"
#include "io/module/module.h"
#include "model/model.h"
#include "state/state.h"
#include "frontend.h"

Environment::Environment(State *state)
        : state(state),
          buffer(build_buffer(
              ResourceManager::get_instance()->get_host_id(), state->model)) {
    // Extract modules
    for (auto& layer : state->model->get_layers()) {
        // Add input module
        Module *input_module = layer->get_input_module();
        if (input_module != nullptr)
            this->input_modules.push_back(input_module);

        // Add expected module
        Module *expected_module = layer->get_expected_module();
        if (expected_module != nullptr)
            this->expected_modules.push_back(expected_module);

        // Add output modules
        for (auto& output_module : layer->get_output_modules())
            this->output_modules.push_back(output_module);
    }
}

Environment::~Environment() {
    delete buffer;
    for (auto& module : this->input_modules) delete module;
    for (auto& module : this->output_modules) delete module;
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
    Frontend::launch_all();
}

void Environment::ui_update() {
    Frontend::update_all(this);
}
