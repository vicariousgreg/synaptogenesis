#include "model/layer.h"
#include "io/module/module.h"
#include "util/error_manager.h"

int Layer::count = 0;

Layer::Layer(Structure *structure, LayerConfig config)
        : name(config.name),
          id(Layer::count++),
          neural_model(config.neural_model),
          structure(structure),
          rows(config.rows),
          columns(config.columns),
          size(rows * columns),
          params(config.params),
          noise(config.noise),
          type(INTERNAL),
          input_module(nullptr),
          dendritic_root(new DendriticNode(0, this)) { }

void Layer::add_input_connection(Connection* connection) {
    this->input_connections.push_back(connection);
}

void Layer::add_output_connection(Connection* connection) {
    this->output_connections.push_back(connection);
}

void Layer::add_to_root(Connection* connection) {
    this->dendritic_root->add_child(connection);
}

void Layer::add_module(Module *module) {
    IOType new_type = module->get_type();

    switch (new_type) {
        case INPUT:
            if (this->input_module != nullptr)
                ErrorManager::get_instance()->log_error(
                    "Layer cannot have more than one input module!");
            this->input_module = module;
            break;
        case OUTPUT:
            this->output_modules.push_back(module);
            break;
        case INPUT_OUTPUT:
            if (this->input_module != nullptr)
                ErrorManager::get_instance()->log_error(
                    "Layer cannot have more than one input module!");
            this->input_module = module;
            this->output_modules.push_back(module);
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unrecognized module type!");
    }

    if (this->input_module != nullptr and this->output_modules.size() > 0)
        this->type = INPUT_OUTPUT;
    else if (this->input_module != nullptr)
        this->type = INPUT;
    else if (this->output_modules.size() > 0)
        this->type = OUTPUT;
    else  // This shouldn't happen, but here it is anyway
        this->type = INTERNAL;
}
