#ifndef column_h
#define column_h

#include "model/structure.h"

class Column : public Structure {
    public:
        Column(std::string name, int cortex_size, bool plastic);

        void add_input(bool plastic, int num_symbols,
            std::string module_name, std::string module_params);

        static void connect(Column *col_a, Column *col_b,
            std::string name_a, std::string name_b, bool reentrant);

    private:
        void add_neural_field(std::string field_name);
        void connect_fields_one_way(std::string src, std::string dest);
        void connect_fields_reentrant(std::string src, std::string dest);
        void add_thalamic_nucleus();
        void add_thalamocortical_reentry(std::string src, std::string dest);

        std::string conductance;

        // Excitatory cortex variables
        int cortex_size;
        bool exc_plastic;

        // Inhibitory cortex variables
        bool exc_inh_plastic;
        std::string exc_self_conductance;
        std::string inh_self_conductance;

        // Thalamus variables
        int thal_size;
        std::string thal_conductance;
};

#endif
