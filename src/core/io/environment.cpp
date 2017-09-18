#include "io/environment.h"

void Environment::add_module(ModuleConfig* config) {
    auto structure = config->get_structure();
    auto layer = config->get_layer();
    auto io_type = io_type_map[structure][layer];
    auto config_type = Module::get_type(config);

    if (config_type & INPUT & io_type)
        ErrorManager::get_instance()->log_error(
            "Error in environment model:\n"
            "  Error adding module to layer: "
                + layer + " in structure: " + structure + "\n"
            "    Layer cannot have more than one input module!");
    if (config_type & EXPECTED & io_type)
        ErrorManager::get_instance()->log_error(
            "Error in environment model:\n"
            "  Error adding module to layer: "
                + layer + " in structure: " + structure + "\n"
            "    Layer cannot have more than one expected module!");

    config_map[structure][layer].push_back(config);
    io_type_map[structure][layer] |= config_type;
    config_list.push_back(config);
}

void Environment::remove_modules() {
    for (auto c : config_list) delete c;
    config_map.clear();
    config_list.clear();
    io_type_map.clear();
}

void Environment::remove_modules(std::string structure,
        std::string layer) {
    for (auto l : config_map[structure]) {
        if (layer == "" or l.first == layer) {
            for (auto c : l.second) delete c;
            l.second.clear();
            io_type_map[structure][l.first] = 0;
        }
    }
}

IOTypeMask Environment::get_type(std::string structure, std::string layer)
    { return io_type_map.at(structure).at(layer); }
