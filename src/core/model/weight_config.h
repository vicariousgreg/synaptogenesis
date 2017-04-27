#ifndef weight_config_h
#define weight_config_h

#include <string>

class Connection;

class WeightConfig {
    public:
        WeightConfig() : diagonal(true) { }
        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

        WeightConfig *set_diagonal(bool diag) {
            diagonal = diag;
            return this;
        }

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
        SurroundWeightConfig(int rows, int cols, WeightConfig* base_config);
        SurroundWeightConfig(int size, WeightConfig* base_config);
        virtual ~SurroundWeightConfig();

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

    private:
        int rows, cols;
        WeightConfig *base_config;
};

class SpecifiedWeightConfig : public WeightConfig {
    public:
        SpecifiedWeightConfig(std::string weight_string)
            : weight_string(weight_string) { }

        virtual void initialize(float* target_matrix,
            Connection* conn, bool is_host);

    private:
        std::string weight_string;
};

#endif
