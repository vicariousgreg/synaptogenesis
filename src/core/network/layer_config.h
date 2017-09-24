#ifndef layer_config_h
#define layer_config_h

#include <string>
#include <map>

#include "util/property_config.h"
#include "util/constants.h"

class NoiseConfig : public PropertyConfig {
    public:
        NoiseConfig(PropertyConfig *config) {
            this->set("type", config->get("type"));
            for (auto pair : config->get())
                this->set(pair.first, pair.second);
        }

        NoiseConfig(std::string type)
            { this->set("type", type); }

        /* Setter that returns self pointer */
        NoiseConfig *set(std::string key, std::string value) {
            set_internal(key, value);
            return this;
        }
};

class LayerConfig : public PropertyConfig {
    public:
        LayerConfig(PropertyConfig *config)
            : name(config->get("name")),
              neural_model(config->get("neural model")),
              rows(std::stoi(config->get("rows", "0"))),
              columns(std::stoi(config->get("columns", "0"))),
              plastic(config->get("plastic", "false") == "true"),
              global(config->get("global", "false") == "true"),
              noise_config(nullptr) {
            for (auto pair : config->get())
                this->set(pair.first, pair.second);
        }

        LayerConfig(
            std::string name,
            std::string neural_model,
            int rows,
            int columns,
            NoiseConfig* noise_config=nullptr,
            bool plastic=false,
            bool global=false)
                : name(name),
                  neural_model(neural_model),
                  rows(rows),
                  columns(columns),
                  noise_config(noise_config),
                  plastic(plastic),
                  global(global) { }

        LayerConfig(
            std::string name,
            std::string neural_model,
            NoiseConfig* noise_config=nullptr,
            bool plastic=false,
            bool global=false)
                : LayerConfig(name, neural_model,
                    0, 0, noise_config, plastic, global) { }

        virtual ~LayerConfig() { delete noise_config; }

        void set_noise_config(NoiseConfig* config)
            { noise_config = noise_config; }
        NoiseConfig* get_noise_config() const { return noise_config; }

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

    protected:
        NoiseConfig* noise_config;
};

#endif
