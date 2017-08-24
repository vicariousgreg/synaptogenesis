#ifndef weight_config_h
#define weight_config_h

#include <string>

#include "model/property_config.h"

class Connection;

class WeightConfig : public PropertyConfig {
    public:
        WeightConfig(std::string type) : diagonal(true) {
            this->set_property("type", type);
        }
        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

        WeightConfig *set_diagonal(bool diag) {
            diagonal = diag;
            return this;
        }

        WeightConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }

        virtual WeightConfig *get_child() { return nullptr; }

    protected:
        bool diagonal;
};

class FlatWeightConfig : public WeightConfig {
    public:
        FlatWeightConfig(float weight, float fraction=1.0);

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

    private:
        float weight;
        float fraction;
};

class RandomWeightConfig : public WeightConfig {
    public:
        RandomWeightConfig(float max_weight, float fraction=1.0);

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

    private:
        float max_weight;
        float fraction;
};

class GaussianWeightConfig : public WeightConfig {
    public:
        GaussianWeightConfig(float mean, float std_dev, float fraction=1.0);

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

    private:
        float mean, std_dev;
        float fraction;
};

class LogNormalWeightConfig : public WeightConfig {
    public:
        LogNormalWeightConfig(float mean, float std_dev, float fraction=1.0);

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

    private:
        float mean, std_dev;
        float fraction;
};

class SurroundWeightConfig : public WeightConfig {
    public:
        SurroundWeightConfig(int rows, int cols, WeightConfig* child_config);
        SurroundWeightConfig(int size, WeightConfig* child_config);
        virtual ~SurroundWeightConfig();

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

        virtual WeightConfig *get_child() { return child_config; }

    private:
        int rows, cols;
        WeightConfig *child_config;
};

class SpecifiedWeightConfig : public WeightConfig {
    public:
        SpecifiedWeightConfig(std::string weight_string)
            : WeightConfig("specified"), weight_string(weight_string) { }

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

    private:
        std::string weight_string;
};

#endif
