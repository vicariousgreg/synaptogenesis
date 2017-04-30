#ifndef column_h
#define column_h

#include "model/structure.h"

class Column : public Structure {
    public:
        Column(std::string name, int rows, int columns, bool plastic);

        void add_input(bool plastic, int num_symbols,
            std::string module_name, std::string module_params);

        static void connect(Column *col_a, Column *col_b,
            std::string name_a, std::string name_b);

    private:
        void add_neural_field(std::string field_name);
        void connect_fields_one_way(std::string src, std::string dest);

        std::string conductance;

        int cortex_rows, cortex_columns;
        int inh_rows, inh_columns;
        bool exc_plastic;
};

#endif
