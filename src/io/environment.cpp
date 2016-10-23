#include "io/environment.h"
#include "io/module.h"

Environment::Environment(Model *model) {
    // Extract modules
    for (int i = 0; i < model->all_layers.size(); ++i) {
        Module *input_module = model->all_layers[i]->input_module;
        Module *output_module = model->all_layers[i]->output_module;
        if (input_module != NULL)
            this->input_modules.push_back(input_module);
        if (output_module != NULL)
            this->output_modules.push_back(output_module);
    }
}

void Environment::step_input(Buffer *buffer) {
    // Run input modules
    for (int i = 0 ; i < this->input_modules.size(); ++i)
        this->input_modules[i]->feed_input(buffer);
}

void Environment::step_output(Buffer *buffer) {
    // Run output modules
    // If no module, skip layer
    for (int i = 0 ; i < this->output_modules.size(); ++i)
        this->output_modules[i]->report_output(buffer);
}

