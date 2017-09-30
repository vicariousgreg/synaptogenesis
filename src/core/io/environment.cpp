#include "io/environment.h"
#include "builder.h"

Environment::Environment(PropertyConfig *config)
        : PropertyConfig(config) {
}

Environment* Environment::load(std::string path) {
    return load_environment(path);
}

void Environment::save(std::string path) {
    save_environment(this, path);
}

void Environment::add_module(PropertyConfig* config) {
    add_to_array("modules", config);
}

void Environment::remove_modules() {
    if (has_array("modules"))
        for (auto config : remove_array("modules"))
            delete config;
}

void Environment::remove_modules(std::string structure,
        std::string layer, std::string type) {
    for (auto config : get_array("modules")) {
        if (config->get("structure", "") == structure and
            (layer == "" or config->get("layer", "") == layer) and
            (type == "" or config->get("type", "") == type))
            delete remove_from_array("modules", config);
    }
}
