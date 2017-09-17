#include "io/environment.h"
#include "io/buffer.h"
#include "io/module/module.h"
#include "state/attributes.h"
#include "model/model.h"
#include "frontend.h"

Environment::Environment(EnvironmentModel *env_model,
        Model* net_model, bool suppress_output) {
    LayerList input_layers, expected_layers, output_layers;

    for (auto config : env_model->get_modules()) {
        if (suppress_output and (Module::get_type(config) & OUTPUT))
            continue;

        Module *module = Module::build_module(net_model, config);
        auto layer = module->layer;
        all_modules.push_back(module);
        auto type = module->get_type();

        if (type & INPUT) this->input_modules.push_back(module);
        if (type & OUTPUT) this->output_modules.push_back(module);
        if (type & EXPECTED) this->expected_modules.push_back(module);
        this->io_types[layer] |= type;
    }

    for (auto pair : io_types) {
        if (pair.second & INPUT)
            input_layers.push_back(pair.first);
        if (pair.second & OUTPUT)
            output_layers.push_back(pair.first);
        if (pair.second & EXPECTED)
            expected_layers.push_back(pair.first);
    }

    buffer = build_buffer(
        ResourceManager::get_instance()->get_host_id(),
            input_layers, output_layers, expected_layers);
}

Environment::~Environment() {
    delete buffer;
    for (auto module : all_modules) delete module;
}

void Environment::step_input() {
    for (auto& module : this->input_modules)
        module->feed_input(buffer);

    for (auto& module : this->expected_modules)
        module->feed_expected(buffer);
}

void Environment::step_output() {
    for (auto& module : this->output_modules)
        module->report_output(buffer,
            Attributes::get_output_type(module->layer));
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
