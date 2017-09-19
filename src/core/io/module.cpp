#include "io/module.h"
#include "network/network.h"
#include "network/layer.h"
#include "util/error_manager.h"

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
    auto layer = network
        ->get_structure(config->get_structure())
        ->get_layer(config->get_layer());
    LayerList layers;
    layers.push_back(layer);

    // Ensure there are layers in the set
    if (layers.size() == 0)
        ErrorManager::get_instance()->log_error(
            "Attempted to build " + type + " module with 0 layers!");

    // Build using structure and layer name
    return bank->build_pointers[type](layers, config);
}

Module::ModuleBank* Module::get_module_bank() {
    static Module::ModuleBank* bank = new ModuleBank();
    return bank;
}

int Module::register_module(std::string module_type,
        IOTypeMask type, MODULE_BUILD_PTR build_ptr) {
    auto bank = Module::get_module_bank();
    if (bank->modules.count(module_type) == 1)
        ErrorManager::get_instance()->log_error(
            "Duplicate module type: " + module_type + "!");
    bank->modules.insert(module_type);
    bank->build_pointers[module_type] = build_ptr;
    bank->io_types[module_type] = type;

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
