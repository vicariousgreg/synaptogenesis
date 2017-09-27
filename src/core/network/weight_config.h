#ifndef weight_config_h
#define weight_config_h

#include <string>

#include "util/property_config.h"
#include "util/error_manager.h"

class Connection;

void initialize_weights(const PropertyConfig config,
    float* target_matrix, Connection* conn, bool is_host);

class FlatWeightConfig : public PropertyConfig {
    public:
        FlatWeightConfig(float weight, float fraction=1.0) {
            this->set_value("type", "flat");
            this->set_value("weight", std::to_string(weight));
            this->set_value("fraction", std::to_string(fraction));
        }
};

class RandomWeightConfig : public PropertyConfig {
    public:
        RandomWeightConfig(float max_weight, float fraction=1.0) {
            this->set_value("type", "random");
            this->set_value("max weight", std::to_string(max_weight));
            this->set_value("fraction", std::to_string(fraction));
        }
};

class GaussianWeightConfig : public PropertyConfig {
    public:
        GaussianWeightConfig(float mean, float std_dev, float fraction=1.0) {
            this->set_value("type", "gaussian");
            this->set_value("mean", std::to_string(mean));
            this->set_value("std dev", std::to_string(std_dev));
            this->set_value("fraction", std::to_string(fraction));
        }
};

class LogNormalWeightConfig : public PropertyConfig {
    public:
        LogNormalWeightConfig(float mean, float std_dev, float fraction=1.0) {
            this->set_value("type", "log normal");
            this->set_value("mean", std::to_string(mean));
            this->set_value("std dev", std::to_string(std_dev));
            this->set_value("fraction", std::to_string(fraction));
        }
};

class SurroundWeightConfig : public PropertyConfig {
    public:
        SurroundWeightConfig(int rows, int cols, PropertyConfig* child_config) {
            for (auto pair : child_config->get())
                this->set_value(pair.first, pair.second);
            this->set_value("type", "surround");
            this->set_value("rows", std::to_string(rows));
            this->set_value("columns", std::to_string(cols));
            this->set_value("child type", child_config->get("type"));
        }

        SurroundWeightConfig(int size, PropertyConfig* child_config) {
            for (auto pair : child_config->get())
                this->set_value(pair.first, pair.second);
            this->set_value("type", "surround");
            this->set_value("size", std::to_string(size));
            this->set_value("child type", child_config->get("type"));
        }
};

class SpecifiedWeightConfig : public PropertyConfig {
    public:
        SpecifiedWeightConfig(std::string weight_string) {
            this->set_value("type", "specified");
            this->set_value("weight string", weight_string);
        }
};

#endif
