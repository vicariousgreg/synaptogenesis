#include "io/module.h"
#include "network/network.h"
#include "network/structure.h"
#include "network/layer.h"
#include "util/error_manager.h"

ModuleConfig::ModuleConfig(std::string type) {
    this->set("type", type);
}

ModuleConfig::ModuleConfig(std::string type,
        std::string structure, std::string layer) {
    this->set("type", type);
    this->add_layer(structure, layer);
}

ModuleConfig* ModuleConfig::add_layer(std::string structure, std::string layer) {
    add_layer(new PropertyConfig(
        { {"structure", structure},
          {"layer", layer} }));
    return this;
}

ModuleConfig* ModuleConfig::add_layer(std::string structure,
        std::string layer, std::string params) {
    add_layer(new PropertyConfig(
        { {"structure", structure},
          {"layer", layer},
          {"params", params} }));
    return this;
}

ModuleConfig* ModuleConfig::add_layer(PropertyConfig *config) {
    if (not config->has_property("structure") or
        not config->has_property("layer"))
    ErrorManager::get_instance()->log_error(
        "Module layer config must have structure and layer name!");
    this->layers.push_back(config);
    this->layer_map[config->get("structure")]
                   [config->get("layer")] = config;
    return this;
}

const PropertyConfig* ModuleConfig::get_layer(Layer *layer) const
    { return layer_map.at(layer->structure->name).at(layer->name); }

Module::Module(LayerList layers) : layers(layers) {
    for (auto layer : layers)
        output_types[layer] = Attributes::get_output_type(layer);
}

Module* Module::build_module(Network *network, ModuleConfig *config) {
    // Check type
    auto type = config->get_type();
    auto bank = Module::get_module_bank();
    if (bank->modules.count(type) == 0)
        ErrorManager::get_instance()->log_error(
            "Unrecognized module: " + type + "!");

    // Extract layers
    LayerList layers;
    for (auto layer_conf : config->get_layers())
        layers.push_back(
            network->get_structure(layer_conf->get("structure"))
                   ->get_layer(layer_conf->get("layer")));

    // Ensure there are layers in the set
    if (layers.size() == 0)
        ErrorManager::get_instance()->log_error(
            "Attempted to build " + type + " module with 0 layers!");

    // Build using structure and layer name
    return bank->build_pointers.at(type)(layers, config);
}

Module::ModuleBank* Module::get_module_bank() {
    static Module::ModuleBank* bank = new ModuleBank();
    return bank;
}

int Module::register_module(std::string module_type,
        MODULE_BUILD_PTR build_ptr) {
    auto bank = Module::get_module_bank();
    if (bank->modules.count(module_type) == 1)
        ErrorManager::get_instance()->log_error(
            "Duplicate module type: " + module_type + "!");
    bank->modules.insert(module_type);
    bank->build_pointers[module_type] = build_ptr;

    // Return index as an identifier
    return bank->modules.size() - 1;
}

void Module::enforce_single_layer(std::string type) {
    if (layers.size() > 1)
        ErrorManager::get_instance()->log_error(
            type + " module only supports a single layer!");
}

void Module::enforce_equal_layer_sizes(std::string type) {
    if (not check_equal_sizes(layers))
        ErrorManager::get_instance()->log_error(
            "Layers in " + type + " module must be of equal sizes!");
}

void Module::set_io_type(IOTypeMask io_type) {
    for (auto layer : layers)
        io_types[layer] = io_type;
}

void Module::set_io_type(Layer *layer, IOTypeMask io_type) {
    io_types[layer] = io_type;
}

OutputType Module::get_output_type(Layer *layer) {
    return output_types.at(layer);
}
