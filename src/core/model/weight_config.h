#ifndef weight_config_h
#define weight_config_h

#include <string>

#include "util/property_config.h"

class Connection;

class WeightConfig : public PropertyConfig {
    public:
        WeightConfig(std::string type) {
            this->set_property("type", type);
        }

        virtual ~WeightConfig()
            { if (child_config != nullptr) delete child_config; }


        void initialize(float* target_matrix, Connection* conn, bool is_host);

        WeightConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }

        void set_child(WeightConfig *child) { child_config = child; }
        WeightConfig *get_child() { return child_config; }

    protected:
        WeightConfig *child_config;

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
            this->set_property("weight", std::to_string(weight));
            this->set_property("fraction", std::to_string(fraction));
        }
};

class RandomWeightConfig : public WeightConfig {
    public:
        RandomWeightConfig(float max_weight, float fraction=1.0)
                : WeightConfig("random") {
            this->set_property("max weight", std::to_string(max_weight));
            this->set_property("fraction", std::to_string(fraction));
        }
};

class GaussianWeightConfig : public WeightConfig {
    public:
        GaussianWeightConfig(float mean, float std_dev, float fraction=1.0)
                : WeightConfig("gaussian") {
            this->set_property("mean", std::to_string(mean));
            this->set_property("std dev", std::to_string(std_dev));
            this->set_property("fraction", std::to_string(fraction));
        }
};

class LogNormalWeightConfig : public WeightConfig {
    public:
        LogNormalWeightConfig(float mean, float std_dev, float fraction=1.0)
                : WeightConfig("log normal") {
            this->set_property("mean", std::to_string(mean));
            this->set_property("std dev", std::to_string(std_dev));
            this->set_property("fraction", std::to_string(fraction));
        }
};

class SurroundWeightConfig : public WeightConfig {
    public:
        SurroundWeightConfig(int rows, int cols, WeightConfig* child_config)
                : WeightConfig("surround") {
            this->set_property("rows", std::to_string(rows));
            this->set_property("columns", std::to_string(cols));
            this->child_config = child_config;
        }

        SurroundWeightConfig(int size, WeightConfig* child_config)
                : WeightConfig("surround") {
            this->set_property("size", std::to_string(size));
            this->child_config = child_config;
        }
};

class SpecifiedWeightConfig : public WeightConfig {
    public:
        SpecifiedWeightConfig(std::string weight_string)
                : WeightConfig("specified") {
            this->set_property("weight string", weight_string);
        }
};

#endif
