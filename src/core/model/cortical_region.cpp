#include "model/cortical_region.h"

CorticalRegion::CorticalRegion(Model *model,
    std::string name, bool plastic,
    int num_columns, int column_rows, int column_cols)
        : name(name), plastic(plastic),
          column_rows(column_rows), column_cols(column_cols),
          base_structure(new Structure(name, PARALLEL)) {
    for (int i = 0 ; i < num_columns ; ++i) {
        std::string column_name = name + std::to_string(columns.size());
        columns.push_back(new Column(
            column_name, column_rows, column_cols, plastic));
    }
    model->add_structure(base_structure);
    for (auto column : columns)
        model->add_structure(column);
}

void CorticalRegion::add_module_all(std::string type, std::string params) {
    base_structure->add_module_all(type, params);
    for (auto column : columns)
        column->add_module_all(type, params);
}

void CorticalRegion::connect(CorticalRegion *other,
        std::string source_layer, std::string dest_layer,
        int num_tethers, int tether_from_size, int tether_to_size,
        float density) {
    for (auto src_column : columns)
        for (auto dest_column : other->columns)
            Column::connect(src_column, dest_column,
                source_layer, dest_layer,
                num_tethers, tether_from_size, tether_to_size,
                density);
}

void CorticalRegion::connect_diffuse(Structure *structure,
        std::string source_layer, Opcode opcode, float weight) {
    for (auto column : columns)
        for (auto layer : column->get_layers())
            Structure::connect(
                structure, source_layer,
                column, layer->name,
                (new ConnectionConfig(false, 0, 1, FULLY_CONNECTED, opcode,
                    new FlatWeightConfig(weight, 1.0)))
                ->set_property("myelinated", "true"));
}
