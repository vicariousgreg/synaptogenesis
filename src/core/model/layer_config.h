#ifndef layer_config_h
#define layer_config_h

#include <string>
#include <map>

#include "model/property_config.h"
#include "util/constants.h"

class LayerConfig : public PropertyConfig {
    public:
        LayerConfig(
            std::string name,
            std::string neural_model,
            int rows,
            int columns,
            float noise_mean=0.0,
            float noise_std_dev=0.0,
            bool plastic=false,
            bool global=false)
                : name(name),
                  neural_model(neural_model),
                  rows(rows),
                  columns(columns),
                  noise_mean(noise_mean),
                  noise_std_dev(noise_std_dev),
                  plastic(plastic),
                  global(global) { }

        LayerConfig(
            std::string name,
            std::string neural_model,
            float noise_mean=0.0,
            float noise_std_dev=0.0,
            bool plastic=false,
            bool global=false)
                : LayerConfig(name, neural_model,
                    0, 0, noise_mean, noise_std_dev, plastic, global) { }

        /* Setter that returns self pointer */
        LayerConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }

        std::string name;
        std::string neural_model;
        int rows, columns;
        float noise_mean, noise_std_dev;
        bool plastic;
        bool global;
};

#endif
