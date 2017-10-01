#ifndef layer_config_h
#define layer_config_h

#include "util/property_config.h"

class LayerConfig : public PropertyConfig {
    public:
        LayerConfig(const PropertyConfig *config);

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

        LayerConfig* add_dendrite(std::string parent,
            PropertyConfig *config);
        LayerConfig* add_dendrite(std::string parent,
            std::string name, bool second_order=false);

        int get_rows() const { return rows; }
        int get_columns() const { return columns; }

        const std::string name;
        const std::string neural_model;
        const bool plastic;
        const bool global;

    protected:
        int rows, columns;
};

#endif
