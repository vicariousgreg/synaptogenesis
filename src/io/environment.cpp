#include "io/environment.h"
#include "io/module.h"

void Environment::step_input(Buffer *buffer) {
    // Run input modules
    for (int i = 0 ; i < this->model->modules.size(); ++i)
        this->model->modules[i]->feed_input(buffer);
}

void Environment::step_output(Buffer *buffer) {
    // Run output modules
    // If no module, skip layer
    for (int i = 0 ; i < this->model->modules.size(); ++i)
        this->model->modules[i]->report_output(buffer);
}

