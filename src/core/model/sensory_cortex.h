#ifndef sensory_cortex_h
#define sensory_cortex_h

#include "model/cortical_region.h"

class SensoryCortex : public CorticalRegion {
    public:
        SensoryCortex(Model *model, std::string name,
            bool plastic, int num_columns,
            int column_rows, int column_cols);

        void add_input(std::string input_name,
            bool plastic_input, int num_symbols,
            std::string module_name, std::string module_params="");
};

#endif
