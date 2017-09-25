#ifndef layer_config_h
#define layer_config_h

#include <string>
#include <map>

#include "util/property_config.h"
#include "util/constants.h"

class LayerConfig : public PropertyConfig {
    public:
        LayerConfig(PropertyConfig *config)
            : name(config->get("name")),
              neural_model(config->get("neural model")),
              rows(std::stoi(config->get("rows", "0"))),
              columns(std::stoi(config->get("columns", "0"))),
              plastic(config->get("plastic", "false") == "true"),
              global(config->get("global", "false") == "true") {
            for (auto pair : config->get())
                this->set(pair.first, pair.second);
            for (auto pair : config->get_children())
                this->set_child(pair.first, pair.second);
        }

        LayerConfig(
                std::string name,
                std::string neural_model,
                int rows,
                int columns,
                PropertyConfig* noise_config=nullptr,
                bool plastic=false,
                bool global=false)
                    : name(name),
                      neural_model(neural_model),
                      rows(rows),
                      columns(columns),
                      plastic(plastic),
                      global(global) {
            if (noise_config != nullptr)
                this->set_child("noise", noise_config);
        }

        LayerConfig(
            std::string name,
            std::string neural_model,
            PropertyConfig* noise_config=nullptr,
            bool plastic=false,
            bool global=false)
                : LayerConfig(name, neural_model,
                    0, 0, noise_config, plastic, global) { }

        /* Setter that returns self pointer */
        LayerConfig *set(std::string key, std::string value) {
            set_internal(key, value);
            return this;
        }

        std::string name;
        std::string neural_model;
        int rows, columns;
        bool plastic;
        bool global;
};

#endif
