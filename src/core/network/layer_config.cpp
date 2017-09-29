#include "network/layer_config.h"
#include "network/layer.h"
#include "util/error_manager.h"

LayerConfig::LayerConfig(PropertyConfig *config)
        : PropertyConfig(config),
          name(config->get("name", "")),
          neural_model(config->get("neural model", "")),
          rows(std::stoi(config->get("rows", "0"))),
          columns(std::stoi(config->get("columns", "0"))),
          plastic(config->get("plastic", "false") == "true"),
          global(config->get("global", "false") == "true") {
    if (not config->has("name"))
        ErrorManager::get_instance()->log_error(
            "Attempted to construct LayerConfig without name!");
    if (not config->has("neural model"))
        ErrorManager::get_instance()->log_error(
            "Attempted to construct LayerConfig without neural model!");
}

LayerConfig::LayerConfig(
        std::string name,
        std::string neural_model,
        int rows,
        int columns,
        PropertyConfig* noise_config,
        bool plastic,
        bool global)
            : name(name),
              neural_model(neural_model),
              rows(rows),
              columns(columns),
              plastic(plastic),
              global(global) {
    if (noise_config != nullptr)
        this->set_child("noise config", noise_config);
}

LayerConfig::LayerConfig(
    std::string name,
    std::string neural_model,
    PropertyConfig* noise_config,
    bool plastic,
    bool global)
        : LayerConfig(name, neural_model,
            0, 0, noise_config, plastic, global) { }

static void add_dendrites_helper(Layer* layer,
        std::string parent_name, const ConfigArray& dendrites) {
    auto parent = layer->get_dendritic_node(parent_name);
    for (auto dendrite : dendrites) {
        if (not dendrite->has("name"))
            ErrorManager::get_instance()->log_error(
                "Attempted to dendrite without name to layer!");

        auto child = parent->add_child(dendrite->get("name"));

        if (dendrite->get("second order", "false") == "true")
            child->set_second_order();

        add_dendrites_helper(layer, child->name,
            dendrite->get_array("children"));
    }
}

void LayerConfig::add_dendrites(Layer* layer) {
    add_dendrites_helper(layer, "root", get_array("dendrites"));
}
