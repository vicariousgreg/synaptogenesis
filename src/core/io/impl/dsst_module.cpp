#include "io/impl/dsst_module.h"

#include "dsst_window.h"

REGISTER_MODULE(DSSTModule, "dsst");

DSSTModule::DSSTModule(LayerList layers, ModuleConfig *config)
        : Module(layers) {
    // Extract primary values, or use defaults
    this->num_cols = config->get_int("columns", 18);
    this->num_rows = config->get_int("rows", 8);
    this->cell_cols = config->get_int("cell size", 8);

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
        auto layer_config = config->get_layer(layer);

        if (layer_config->get_bool("input", false))
            set_io_type(layer, get_io_type(layer) | INPUT);

        if (layer_config->get_bool("expected", false))
            set_io_type(layer, get_io_type(layer) | EXPECTED);

        if (layer_config->get_bool("output", false))
            set_io_type(layer, get_io_type(layer) | OUTPUT);

        // Use output as default
        if (get_io_type(layer) == 0)
            set_io_type(layer, INPUT);
        window->add_layer(layer, get_io_type(layer));
    }
}

void DSSTModule::feed_input(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & INPUT)
            window->feed_input(layer, buffer->get_input(layer));
}

void DSSTModule::report_output(Buffer *buffer) {
    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT) ;
            // Add output processing here
}

void DSSTModule::input_symbol(int index) {
    window->input_symbol(index);
}

int DSSTModule::get_num_rows(PropertyConfig *config) {
    return config->get_int("rows", 8);
}

int DSSTModule::get_num_columns(PropertyConfig *config) {
    return config->get_int("columns", 18);
}

int DSSTModule::get_cell_rows(PropertyConfig *config) {
    return 1+2*get_cell_columns(config);
}

int DSSTModule::get_cell_columns(PropertyConfig *config) {
    return config->get_int("cell size", 8);
}

int DSSTModule::get_cell_size(PropertyConfig *config) {
    return get_cell_rows(config) * get_cell_columns(config);
}

int DSSTModule::get_spacing(PropertyConfig *config) {
    return get_cell_columns(config) / 4;
}

int DSSTModule::get_input_rows(PropertyConfig *config) {
    int num_rows = get_num_rows(config);
    int cell_rows = get_cell_rows(config);
    int spacing = get_spacing(config);
    return (num_rows + 2) * (cell_rows + spacing) - spacing;
}

int DSSTModule::get_input_columns(PropertyConfig *config) {
    int num_cols = get_num_columns(config);
    int cell_cols = get_cell_columns(config);
    int spacing = get_spacing(config);
    return num_cols * (cell_cols + spacing) - spacing;
}

int DSSTModule::get_input_size(PropertyConfig *config) {
    return get_input_rows(config) * get_input_columns(config);
}
