#include "model/layer_config.h"

std::string LayerConfig::get_property(std::string key) {
    return properties.at(key);
}

LayerConfig *LayerConfig::set_property(
        std::string key, std::string value) {
    properties[key] = value;
    return this;
}
