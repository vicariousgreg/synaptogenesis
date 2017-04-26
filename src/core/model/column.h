#ifndef column_h
#define column_h

#include "model/structure.h"

class Column : public Structure {
    public:
        Column::Column(std::string name, int cortex_size);

        static void connect(Column *col_a, Column *col_b,
            std::string name_a, std::string name_b);

    private:
        void add_neural_field(std::string field_name);
        void connect_fields_one_way(std::string src, std::string dest);
        void connect_fields_reentrant(std::string src, std::string dest);
        void add_thalamic_nucleus();
        void add_thalamocortical_reentry(std::string src, std::string dest);

        // Global variables
        std::string conductance;
        std::string learning_rate;
        std::string reentrant_learning_rate;

        // Excitatory cortex variables
        int cortex_size;
        float exc_noise_mean;
        float exc_noise_std_dev;
        bool exc_plastic;

        // Inhibitory cortex variables
        int inh_ratio;
        int inh_size;
        float inh_noise_mean;
        float inh_noise_std_dev;
        bool exc_inh_plastic;
        int exc_inh_spread;
        int inh_exc_spread;
        std::string exc_inh_conductance;
        std::string inh_exc_conductance;

        // Intracortical connection variables
        int intracortical_delay;

        // Thalamus variables
        int thal_ratio;
        int thal_size;
        int thal_spread;
        float thal_noise_mean;
        float thal_noise_std_dev;
        int thalamocortical_delay;
};

#endif
