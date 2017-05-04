#ifndef motor_cortex_h
#define motor_cortex_h

#include "model/cortical_region.h"

class MotorCortex : public CorticalRegion {
    public:
        MotorCortex(Model *model, std::string name,
            bool plastic, int num_columns,
            int column_rows, int column_cols);

        void add_output(std::string output_name,
            bool plastic_output, int num_symbols,
            std::string module_name, std::string module_params="");
};

#endif
