#include "io/environment.h"
#include "io/input.h"
#include "io/output.h"

void Environment::step_input(Buffer *buffer) {
    // Run input modules
    for (int i = 0 ; i < this->model->input_modules.size(); ++i)
        this->model->input_modules[i]->feed_input(buffer);
}

void Environment::step_output(Buffer *buffer) {
    // Run output modules
    // If no module, skip layer
    for (int i = 0 ; i < this->model->output_modules.size(); ++i)
        this->model->output_modules[i]->report_output(buffer);
}

