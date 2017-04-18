#include "model/layer.h"
#include "io/module/module.h"
#include "util/error_manager.h"

int Layer::count = 0;

Layer::Layer(Structure *structure, LayerConfig *config)
        : config(config),
          name(config->name),
          id(Layer::count++),
          neural_model(config->neural_model),
          structure(structure),
          rows(config->rows),
          columns(config->columns),
          size(rows * columns),
          noise(config->noise),
          plastic(config->plastic),
          global(config->global),
          type(0),
          input_module(nullptr),
          expected_module(nullptr),
          dendritic_root(new DendriticNode(0, this)) { }

Layer::~Layer() {
    delete dendritic_root;
    delete config;
}

IOTypeMask Layer::get_type() const { return type; }
bool Layer::is_input() const { return type & INPUT; }
bool Layer::is_output() const { return type & OUTPUT; }
bool Layer::is_expected() const { return type & EXPECTED; }

Module* Layer::get_input_module() const { return input_module; }
Module* Layer::get_expected_module() const { return expected_module; }
const ModuleList Layer::get_output_modules() const { return output_modules; }

const ConnectionList& Layer::get_input_connections() const {
    return input_connections; }
const ConnectionList& Layer::get_output_connections() const {
    return output_connections;
}

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
