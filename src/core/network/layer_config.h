#ifndef layer_config_h
#define layer_config_h

#include <string>

#include "util/property_config.h"

class Layer;

class LayerConfig : public PropertyConfig {
    public:
        LayerConfig(PropertyConfig *config);

        LayerConfig(
                std::string name,
                std::string neural_model,
                int rows,
                int columns,
                PropertyConfig* noise_config=nullptr,
                bool plastic=false,
                bool global=false);

        LayerConfig(
            std::string name,
            std::string neural_model,
            PropertyConfig* noise_config=nullptr,
            bool plastic=false,
            bool global=false);

        void add_dendrites(Layer* layer);

        /* Setter that returns self pointer */
        LayerConfig *set(std::string key, std::string value) {
            set_value(key, value);
            return this;
        }

        const std::string name;
        const std::string neural_model;
        int rows, columns;
        const bool plastic;
        const bool global;
};

#endif
