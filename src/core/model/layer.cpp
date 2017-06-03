#include "model/layer.h"
#include "model/connection.h"
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
          noise_mean(config->noise_mean),
          noise_std_dev(config->noise_std_dev),
          plastic(config->plastic),
          global(config->global),
          type(0),
          dendritic_root(new DendriticNode(0, this)) { }

Layer::~Layer() {
    delete dendritic_root;
    delete config;
    for (auto config : module_configs) delete config;
}

const LayerConfig *Layer::get_config() const { return config; }
IOTypeMask Layer::get_type() const { return type; }
bool Layer::is_input() const { return type & INPUT; }
bool Layer::is_output() const { return type & OUTPUT; }
bool Layer::is_expected() const { return type & EXPECTED; }

const std::vector<ModuleConfig*> Layer::get_module_configs() const
    { return module_configs; }

const ConnectionList& Layer::get_input_connections() const
    { return input_connections; }
const ConnectionList& Layer::get_output_connections() const
    { return output_connections; }

int Layer::get_max_delay() const {
    // Determine max delay for output connections
    int max_delay = 0;
    for (auto& conn : get_output_connections()) {
        int delay = conn->delay;
        if (delay > max_delay)
            max_delay = delay;
    }
    return max_delay;
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

void Layer::add_module(std::string module_name, std::string params) {
    IOTypeMask model_type = Module::get_module_type(module_name);

    if ((model_type & INPUT) and (this->type & INPUT))
        ErrorManager::get_instance()->log_error(
            "Layer cannot have more than one input module!");
    if ((model_type & EXPECTED) and (this->type & EXPECTED))
        ErrorManager::get_instance()->log_error(
            "Layer cannot have more than one expected module!");

    this->module_configs.push_back(new ModuleConfig(module_name, params));

    this->type |= model_type;
}
