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
            this->set("type", "flat");
            this->set("weight", std::to_string(weight));
            this->set("fraction", std::to_string(fraction));
        }
};

class RandomWeightConfig : public PropertyConfig {
    public:
        RandomWeightConfig(float max_weight, float fraction=1.0) {
            this->set("type", "random");
            this->set("max weight", std::to_string(max_weight));
            this->set("fraction", std::to_string(fraction));
        }
};

class GaussianWeightConfig : public PropertyConfig {
    public:
        GaussianWeightConfig(float mean, float std_dev, float fraction=1.0) {
            this->set("type", "gaussian");
            this->set("mean", std::to_string(mean));
            this->set("std dev", std::to_string(std_dev));
            this->set("fraction", std::to_string(fraction));
        }
};

class LogNormalWeightConfig : public PropertyConfig {
    public:
        LogNormalWeightConfig(float mean, float std_dev, float fraction=1.0) {
            this->set("type", "log normal");
            this->set("mean", std::to_string(mean));
            this->set("std dev", std::to_string(std_dev));
            this->set("fraction", std::to_string(fraction));
        }
};

class SurroundWeightConfig : public PropertyConfig {
    public:
        SurroundWeightConfig(int rows, int cols, PropertyConfig* child_config) {
            for (auto pair : child_config->get())
                this->set(pair.first, pair.second);
            this->set("type", "surround");
            this->set("rows", std::to_string(rows));
            this->set("columns", std::to_string(cols));
            if (not child_config->has("type"))
                ErrorManager::get_instance()->log_error(
                    "Attempted to build surround weight config"
                    " without specified child config type!");
            this->set("child type", child_config->get("type"));
        }

        SurroundWeightConfig(int size, PropertyConfig* child_config) {
            for (auto pair : child_config->get())
                this->set(pair.first, pair.second);
            this->set("type", "surround");
            this->set("size", std::to_string(size));
            if (not child_config->has("type"))
                ErrorManager::get_instance()->log_error(
                    "Attempted to build surround weight config"
                    " without specified child config type!");
            this->set("child type", child_config->get("type"));
        }
};

class SpecifiedWeightConfig : public PropertyConfig {
    public:
        SpecifiedWeightConfig(std::string weight_string) {
            this->set("type", "specified");
            this->set("weight string", weight_string);
        }
};

#endif
