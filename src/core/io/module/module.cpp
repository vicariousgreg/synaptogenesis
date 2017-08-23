#include "io/module/module.h"
#include "util/error_manager.h"

Module* Module::build_module(Layer *layer, ModuleConfig *config) {
    auto bank = Module::get_module_bank();
    if (bank->modules.count(config->get_property("name")) == 0)
        ErrorManager::get_instance()->log_error(
            "Unrecognized module: " + config->get_property("name") + "!");

    return bank->build_pointers[config->get_property("name")](layer, config);
}

Module::ModuleBank* Module::get_module_bank() {
    static Module::ModuleBank* bank = new ModuleBank();
    return bank;
}

// Get the IOType of a module subclass
IOTypeMask Module::get_module_type(std::string module_name) {
    try {
        return Module::get_module_bank()->types.at(module_name);
    } catch (...) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized module: " + module_name + "!");
    }
}

IOTypeMask Module::get_module_type(ModuleConfig *config) {
    return Module::get_module_type(config->get_property("name"));
}

int Module::register_module(std::string module_name,
        IOTypeMask type, MODULE_BUILD_PTR build_ptr) {
    auto bank = Module::get_module_bank();
    if (bank->modules.count(module_name) == 1)
        ErrorManager::get_instance()->log_error(
            "Duplicate module name: " + module_name + "!");
    bank->modules.insert(module_name);
    bank->build_pointers[module_name] = build_ptr;
    bank->types[module_name] = type;

    // Return index as an identifier
    return bank->modules.size() - 1;
}
