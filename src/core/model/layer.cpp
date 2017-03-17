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
          plastic(config.plastic),
          type(0),
          input_module(nullptr),
          expected_module(nullptr),
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

#include <iostream>

void Layer::add_module(Module *module) {
    IOTypeMask model_type = module->get_type();

    if ((model_type & INPUT) and (this->type & INPUT))
        ErrorManager::get_instance()->log_error(
            "Layer cannot have more than one input module!");
    if ((model_type & EXPECTED) and (this->type & EXPECTED))
        ErrorManager::get_instance()->log_error(
            "Layer cannot have more than one expected module!");

    if (model_type & INPUT) this->input_module = module;
    if (model_type & OUTPUT) this->output_modules.push_back(module);
    if (model_type & EXPECTED) this->expected_module = module;

    this->type |= model_type;
}
