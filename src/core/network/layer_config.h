#ifndef layer_config_h
#define layer_config_h

#include "util/property_config.h"

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

        LayerConfig* add_dendrite(std::string parent,
            PropertyConfig *config);
        LayerConfig* add_dendrite(std::string parent,
            std::string name, bool second_order=false);

        int get_rows() const { return rows; }
        int get_columns() const { return columns; }
        LayerConfig* set_rows(int x)
            { rows = x; return this; }
        LayerConfig* set_columns(int x)
            { columns = x; return this; }

        /* Setter that returns self pointer */
        LayerConfig *set(std::string key, std::string value) {
            PropertyConfig::set(key, value);
            return this;
        }
        LayerConfig *set_child(std::string key, PropertyConfig* child) {
            PropertyConfig::set_child(key, child);
            return this;
        }
        LayerConfig *set_array(std::string key, ConfigArray array) {
            PropertyConfig::set_array(key, array);
            return this;
        }
        LayerConfig *add_to_array(std::string key, PropertyConfig* config) {
            PropertyConfig::add_to_array(key, config);
            return this;
        }

        const std::string name;
        const std::string neural_model;
        const bool plastic;
        const bool global;

    protected:
        int rows, columns;
};

#endif
