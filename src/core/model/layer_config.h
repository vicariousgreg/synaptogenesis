#ifndef layer_config_h
#define layer_config_h

#include <string>
#include <map>

#include "model/property_config.h"
#include "util/constants.h"

typedef enum NoiseType {
    NORMAL,
    POISSON
} NoiseType;

class NoiseConfig : public PropertyConfig {
    public:
        NoiseConfig(NoiseType type) : type(type) { }
        NoiseType type;

        /* Setter that returns self pointer */
        NoiseConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }
};

class LayerConfig : public PropertyConfig {
    public:
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

        /* Setter that returns self pointer */
        LayerConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }

        std::string name;
        std::string neural_model;
        int rows, columns;
        bool plastic;
        bool global;
        NoiseConfig* const noise_config;
};

#endif
