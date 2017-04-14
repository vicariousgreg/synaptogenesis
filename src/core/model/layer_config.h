#ifndef layer_config_h
#define layer_config_h

#include <string>
#include "util/constants.h"

class LayerConfig {
    public:
        LayerConfig(
            std::string name,
            std::string neural_model,
            int rows,
            int columns,
            std::string params="",
            float noise=0.0,
            bool plastic=false)
                : name(name),
                  neural_model(neural_model),
                  rows(rows),
                  columns(columns),
                  params(params),
                  noise(noise),
                  plastic(plastic){ }

        LayerConfig(
            std::string name,
            std::string neural_model,
            std::string params="",
            float noise=0.0,
            bool plastic=false)
                : LayerConfig(name, neural_model,
                    0, 0, params, noise, plastic) { }
        
        std::string name;
        std::string neural_model;
        int rows, columns;
        std::string params;
        float noise;
        bool plastic;
};

#endif
