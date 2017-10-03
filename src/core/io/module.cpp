#include "io/module.h"
#include "network/network.h"
#include "network/structure.h"
#include "network/layer.h"
#include "util/error_manager.h"

ModuleConfig::ModuleConfig(PropertyConfig *config)
    : PropertyConfig(config) { }

ModuleConfig::ModuleConfig(std::string type) {
    this->set("type", type);
}

ModuleConfig::ModuleConfig(std::string type,
        std::string structure, std::string layer) {
    this->set("type", type);
    this->add_layer(structure, layer);
}

ModuleConfig* ModuleConfig::add_layer(std::string structure,
        std::string layer, std::string params) {
    auto props = new PropertyConfig(
        { {"structure", structure},
          {"layer", layer} });
    if (params != "") props->set("params", params);
    add_layer(props);
    delete props;
    return this;
}

ModuleConfig* ModuleConfig::add_layer(PropertyConfig *config) {
    if (not config->has("structure") or not config->has("layer"))
        ErrorManager::get_instance()->log_error(
            "Module layer config must have structure and layer name!");
    this->add_to_array("layers", config);
    return this;
}

const PropertyConfig* ModuleConfig::get_layer(Layer *layer) const {
    for (auto config : get_array("layers"))
        if (config->get("structure") == layer->structure->name and
            config->get("layer") == layer->name)
            return config;
}

Module::Module(LayerList layers) : layers(layers) {
    for (auto layer : layers)
        output_types[layer] = Attributes::get_output_type(layer);
}

IOTypeMask Module::get_io_type(Layer *layer) const {
    try {
        return io_types.at(layer);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Attempted to retrieve IO type from Module for "
            "unrepresented layer: " + layer->str());
    }
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
    try {
        return output_types.at(layer);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Attempted to retrieve output type from Module for "
            "unrepresented layer: " + layer->str());
    }
}
