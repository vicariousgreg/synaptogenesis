#include "model/sensory_cortex.h"
#include "util/error_manager.h"

#define IZHIKEVICH "izhikevich"
#define RELAY "relay"
#define IZ_INIT "init"

SensoryCortex::SensoryCortex(Model *model,
    bool plastic, int num_columns,
    int column_rows, int column_cols)
        : CorticalRegion(model, "sensory cortex", plastic,
              num_columns, column_rows, column_cols) {
}

void SensoryCortex::add_input(std::string input_name,
        bool plastic_input, int num_symbols,
        std::string module_name, std::string module_params) {
    if (num_symbols > columns.size())
        ErrorManager::get_instance()->log_error(
            "Not enough columns in sensory cortex for output module!");

    base_structure->add_layer(
        (new LayerConfig(input_name, IZHIKEVICH, 1, num_symbols))
        ->set_property(IZ_INIT, "regular"));
    base_structure->add_module(input_name, module_name, module_params);

    for (int i = 0 ; i < num_symbols ; ++i) {
        Column *column = columns[i];

        Structure::connect(
            base_structure, input_name,
            column, "4_pos",
            (new ConnectionConfig(plastic_input, 0, 1, SUBSET, ADD,
            new FlatWeightConfig(0.5, 0.09)))
            ->set_subset_config(
                new SubsetConfig(
                    0, 1,
                    i, i+1,
                    0, column_rows,
                    0, column_cols))
            ->set_property("myelinated", "true"));
    }
}
