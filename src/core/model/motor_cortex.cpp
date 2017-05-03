#include "model/motor_cortex.h"
#include "util/error_manager.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

MotorCortex::MotorCortex(Model *model,
    bool plastic, int num_columns,
    int column_rows, int column_cols)
        : CorticalRegion(model, "motor cortex", plastic,
              num_columns, column_rows, column_cols) {
}

void MotorCortex::add_output(std::string output_name,
        bool plastic_output, int num_symbols,
        std::string module_name, std::string module_params) {
    if (num_symbols > columns.size())
        ErrorManager::get_instance()->log_error(
            "Not enough columns in motor cortex for output module!");

    base_structure->add_layer(
        (new LayerConfig(output_name, IZHIKEVICH, 1, num_symbols))
        ->set_property(IZ_INIT, "bursting"));
    base_structure->add_module(output_name, module_name, module_params);

    for (int i = 0 ; i < num_symbols ; ++i) {
        Column *column = columns[i];

        Structure::connect(
            column, "5_pos",
            base_structure, output_name,
            (new ConnectionConfig(plastic_output, 0, 1, SUBSET, ADD,
            new FlatWeightConfig(0.5, 1.0)))
            ->set_subset_config(
                new SubsetConfig(
                    0, column_rows,
                    0, column_cols,
                    0, 1,
                    i, i+1))
            ->set_property("myelinated", "true"));
    }
}
