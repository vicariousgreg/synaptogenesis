#include "io/module/module.h"
#include "util/error_manager.h"

Module* Module::build_module(Layer *layer, ModuleConfig *config) {
    auto bank = Module::get_module_bank();
    if (bank->modules.count(config->name) == 0)
        ErrorManager::get_instance()->log_error(
            "Unrecognized module string: " + config->name + "!");

    return bank->build_pointers[config->name](layer, config->params);
}

Module::ModuleBank* Module::get_module_bank() {
    static Module::ModuleBank* bank = new ModuleBank();
    return bank;
}

// Get the IOType of a module subclass
IOTypeMask Module::get_module_type(std::string module_name) {
    return Module::get_module_bank()->types.at(module_name);
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
