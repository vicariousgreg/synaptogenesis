#ifndef layer_config_h
#define layer_config_h

#include <string>
#include <map>

#include "util/constants.h"

class LayerConfig {
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

        /* Getter and setter for generic properties
         * Setter returns self pointer for convenience */
        std::string get_property(std::string key);
        LayerConfig *set_property(std::string key, std::string value);
        
        std::string name;
        std::string neural_model;
        int rows, columns;
        float noise_mean, noise_std_dev;
        bool plastic;
        bool global;

    private:
        std::map<std::string, std::string> properties;
};

#endif
