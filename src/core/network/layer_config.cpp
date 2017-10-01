#include "network/layer_config.h"
#include "network/layer.h"
#include "util/error_manager.h"

#define cimg_display 0
#include "CImg.h"

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
    if (config->has("image")) {
        cimg_library::CImg<unsigned char> img(config->get("image").c_str());
        this->rows = img.height();
        this->columns = img.width();
    }
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
    this->set("name", name);
    this->set("neural model", neural_model);
    this->set("rows", std::to_string(rows));
    this->set("columns", std::to_string(columns));
    if (noise_config != nullptr)
        this->set_child("noise config", noise_config);
    this->set("plastic", (plastic) ? "true" : "false");
    this->set("global", (global) ? "true" : "false");
}

LayerConfig::LayerConfig(
    std::string name,
    std::string neural_model,
    PropertyConfig* noise_config,
    bool plastic,
    bool global)
        : LayerConfig(name, neural_model,
            0, 0, noise_config, plastic, global) { }

static PropertyConfig* get_dendrite(const ConfigArray& arr, std::string name) {
    for (auto node : arr)
        if (node->get("name") == name)
            return node;
        else {
            auto props = get_dendrite(node->get_array("children"), name);
            if (props != nullptr) return props;
        }
    return nullptr;
}

LayerConfig* LayerConfig::add_dendrite(std::string parent,
        PropertyConfig *config) {
    if (parent == "root") {
        this->add_to_array("dendrites", config);
    } else {
        auto node = get_dendrite(this->get_array("dendrites"), parent);
        if (node == nullptr)
            ErrorManager::get_instance()->log_error(
                "Could not find dendrite " + parent
                + "in layer " + name + "!");
        node->add_to_array("children", config);
    }
    return this;
}

LayerConfig* LayerConfig::add_dendrite(std::string parent,
            std::string name, bool second_order) {
    auto props = new PropertyConfig();
    props->set("name", name);
    props->set("second order", second_order ? "true" : "false");
    this->add_dendrite(parent, props);
    delete props;
    return this;
}