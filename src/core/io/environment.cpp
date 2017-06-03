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
        for (auto config : layer->get_module_configs()) {
            Module *module = Module::build_module(layer, config);
            auto type = module->get_type();
            if (type & INPUT) this->input_modules.push_back(module);
            if (type & OUTPUT) this->output_modules.push_back(module);
            if (type & EXPECTED) this->expected_modules.push_back(module);
        }
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

void Environment::ui_init() {
    Frontend::init_all();
}

void Environment::ui_launch() {
    Frontend::launch_all();
}

void Environment::ui_update() {
    Frontend::update_all(this);
}
