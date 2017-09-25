#ifndef weight_config_h
#define weight_config_h

#include <string>

#include "util/property_config.h"
#include "util/error_manager.h"

class Connection;

class WeightConfig : public PropertyConfig {
    public:
        WeightConfig() { }

        WeightConfig(PropertyConfig *config) {
            for (auto pair : config->get())
                this->set(pair.first, pair.second);
        }

        WeightConfig(std::string type) {
            this->set("type", type);
        }

        void initialize(float* target_matrix, Connection* conn, bool is_host);

        WeightConfig *set(std::string key, std::string value) {
            set_internal(key, value);
            return this;
        }

    protected:
        void flat_config(float* target_matrix,
            Connection* conn, bool is_host);

        void random_config(float* target_matrix,
            Connection* conn, bool is_host);

        void gaussian_config(float* target_matrix,
            Connection* conn, bool is_host);

        void log_normal_config(float* target_matrix,
            Connection* conn, bool is_host);

        void surround_config(float* target_matrix,
            Connection* conn, bool is_host);

        void specified_config(float* target_matrix,
            Connection* conn, bool is_host);

};

class FlatWeightConfig : public WeightConfig {
    public:
        FlatWeightConfig(float weight, float fraction=1.0)
                : WeightConfig("flat") {
            this->set("weight", std::to_string(weight));
            this->set("fraction", std::to_string(fraction));
        }
};

class RandomWeightConfig : public WeightConfig {
    public:
        RandomWeightConfig(float max_weight, float fraction=1.0)
                : WeightConfig("random") {
            this->set("max weight", std::to_string(max_weight));
            this->set("fraction", std::to_string(fraction));
        }
};

class GaussianWeightConfig : public WeightConfig {
    public:
        GaussianWeightConfig(float mean, float std_dev, float fraction=1.0)
                : WeightConfig("gaussian") {
            this->set("mean", std::to_string(mean));
            this->set("std dev", std::to_string(std_dev));
            this->set("fraction", std::to_string(fraction));
        }
};

class LogNormalWeightConfig : public WeightConfig {
    public:
        LogNormalWeightConfig(float mean, float std_dev, float fraction=1.0)
                : WeightConfig("log normal") {
            this->set("mean", std::to_string(mean));
            this->set("std dev", std::to_string(std_dev));
            this->set("fraction", std::to_string(fraction));
        }
};

class SurroundWeightConfig : public WeightConfig {
    public:
        SurroundWeightConfig(int rows, int cols, WeightConfig* child_config)
                : WeightConfig("surround") {
            this->set("rows", std::to_string(rows));
            this->set("columns", std::to_string(cols));
            this->set("child type", child_config->get("type"));
        }

        SurroundWeightConfig(int size, WeightConfig* child_config)
                : WeightConfig("surround") {
            this->set("size", std::to_string(size));
            this->set("child type", child_config->get("type"));
        }
};

class SpecifiedWeightConfig : public WeightConfig {
    public:
        SpecifiedWeightConfig(std::string weight_string)
                : WeightConfig("specified") {
            this->set("weight string", weight_string);
        }
};

#endif
