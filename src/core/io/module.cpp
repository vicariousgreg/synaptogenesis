#include "io/module.h"
#include "network/network.h"
#include "network/layer.h"
#include "util/error_manager.h"

Module* Module::build_module(Network *network, ModuleConfig *config) {
    // Check type
    auto type = config->get_type();
    auto bank = Module::get_module_bank();
    if (bank->modules.count(type) == 0)
        ErrorManager::get_instance()->log_error(
            "Unrecognized module: " + type + "!");

    // Build using structure and layer name
    return bank->build_pointers[type](
        network->get_structure(config->get_structure())
            ->get_layer(config->get_layer()),
        config);
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
