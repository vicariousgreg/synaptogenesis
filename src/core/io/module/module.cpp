#include "io/module/module.h"
#include "util/error_manager.h"

Module* Module::build_module(Layer *layer, ModuleConfig *config) {
    auto bank = Module::get_module_bank();
    if (bank->modules.count(config->get_property("type")) == 0)
        ErrorManager::get_instance()->log_error(
            "Unrecognized module: " + config->get_property("type") + "!");

    return bank->build_pointers[config->get_property("type")](layer, config);
}

Module::ModuleBank* Module::get_module_bank() {
    static Module::ModuleBank* bank = new ModuleBank();
    return bank;
}

// Get the IOType of a module subclass
IOTypeMask Module::get_module_type(std::string module_type) {
    try {
        return Module::get_module_bank()->io_types.at(module_type);
    } catch (...) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized module: " + module_type + "!");
    }
}

IOTypeMask Module::get_module_type(ModuleConfig *config) {
    return Module::get_module_type(config->get_property("type"));
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
