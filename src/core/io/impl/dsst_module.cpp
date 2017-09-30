#include "io/impl/dsst_module.h"
#include "util/error_manager.h"
#include "dsst_window.h"

REGISTER_MODULE(DSSTModule, "dsst");

DSSTModule::DSSTModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    // Extract primary values, or use defaults
    this->num_cols = std::stoi(config->get("columns", "18"));
    this->num_rows = std::stoi(config->get("rows", "8"));
    this->cell_cols = std::stoi(config->get("cell size", "8"));

    // Set secondary values
    this->cell_rows = 1+2*cell_cols;
    this->cell_size = cell_rows * cell_cols;
    this->spacing = cell_cols/4;
    this->input_rows = (num_rows + 2) * (cell_rows + spacing) - spacing;
    this->input_cols = num_cols * (cell_cols + spacing) - spacing;
    this->input_size = input_rows * input_cols;

    input_data = Pointer<float>(input_size);
    this->window = DSSTWindow::build(this);

    for (auto layer : layers) {
        auto param = config->get_layer(layer)->get("params", "input");
        params[layer] = param;

        if (param == "input") {
            set_io_type(layer, INPUT);
            window->add_layer(layer, INPUT);
        } else if (param == "output") {
            set_io_type(layer, OUTPUT);
            window->add_layer(layer, OUTPUT);
        } else {
            ErrorManager::get_instance()->log_error(
                "Unrecognized layer type: " + param
                + " in VisualizerModule!");
        }
    }
}

void DSSTModule::feed_input(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & INPUT) {
            window->feed_input(layer, buffer->get_input(layer));
            buffer->set_dirty(layer, true);
        }
    }
}

void DSSTModule::report_output(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT) ;
            // Add output processing here
}

void DSSTModule::input_symbol(int index) {
    window->input_symbol(index);
}

int DSSTModule::get_num_rows(ModuleConfig *config) {
    return std::stoi(config->get("rows", "8"));
}

int DSSTModule::get_num_columns(ModuleConfig *config) {
    return std::stoi(config->get("columns", "18"));
}

int DSSTModule::get_cell_rows(ModuleConfig *config) {
    return 1+2*get_cell_columns(config);
}

int DSSTModule::get_cell_columns(ModuleConfig *config) {
    return std::stoi(config->get("cell size", "8"));
}

int DSSTModule::get_cell_size(ModuleConfig *config) {
    return get_cell_rows(config) * get_cell_columns(config);
}

int DSSTModule::get_spacing(ModuleConfig *config) {
    return get_cell_columns(config) / 4;
}

int DSSTModule::get_input_rows(ModuleConfig *config) {
    int num_rows = get_num_rows(config);
    int cell_rows = get_cell_rows(config);
    int spacing = get_spacing(config);
    return (num_rows + 2) * (cell_rows + spacing) - spacing;
}

int DSSTModule::get_input_columns(ModuleConfig *config) {
    int num_cols = get_num_columns(config);
    int cell_cols = get_cell_columns(config);
    int spacing = get_spacing(config);
    return num_cols * (cell_cols + spacing) - spacing;
}

int DSSTModule::get_input_size(ModuleConfig *config) {
    return get_input_rows(config) * get_input_columns(config);
}
