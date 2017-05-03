#ifndef cortical_region_h
#define cortical_region_h

#include <vector>
#include <string>

#include "model/model.h"
#include "model/column.h"

class CorticalRegion {
    public:
        CorticalRegion(Model *model, std::string name, bool plastic,
            int num_columns, int column_rows, int column_cols);

        void add_module_all(std::string type, std::string params="");
        void connect(CorticalRegion *other,
            std::string source_layer, std::string dest_layer,
            int num_tethers, int tether_from_size, int tether_to_size,
            float density);
        void connect_diffuse(Structure *structure,
            std::string source_layer, Opcode opcode, float weight);

    protected:
        const std::string name;
        const bool plastic;
        const int column_rows;
        const int column_cols;
        Structure* const base_structure;
        std::vector<Column*> columns;
};

#endif
