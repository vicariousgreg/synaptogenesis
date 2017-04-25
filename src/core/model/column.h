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

        // Excitatory cortex variables
        int cortex_size;
        float exc_noise_mean;
        float exc_noise_std_dev;
        bool exc_plastic;

        // Inhibitory cortex variables
        int inh_ratio;
        int inh_size;
        bool inh_plastic;

        // Intracortical connection variables
        int spread_ratio;
        int spread_factor;
        int spread;
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
