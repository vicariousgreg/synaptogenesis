#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

#include "io/input.h"
#include "io/output.h"

void Driver::step_input(Buffer *buffer) {
    // Run input modules
    // If no module, clear the input
    for (int i = 0 ; i < this->model->input_modules.size(); ++i)
        this->model->input_modules[i]->feed_input(buffer);
    this->state->copy_input(buffer);
}

void Driver::step_connections() {
    // Calculate inputs for connections
    for (int cid = 0 ; cid < this->model->connections.size(); ++cid)
        step_connection(this->model->connections[cid]);
}

void Driver::step_output(Buffer *buffer) {
    this->state->copy_output(buffer);

    // Run output modules
    // If no module, skip layer
    for (int i = 0 ; i < this->model->output_modules.size(); ++i)
        this->model->output_modules[i]->report_output(buffer);
}

Driver* build_driver(Model* model) {
    Driver* driver;
    if (model->driver_string == "izhikevich")
        driver = new IzhikevichDriver();
    else if (model->driver_string == "rate_encoding")
        driver = new RateEncodingDriver();
    else
        throw "Unrecognized driver type!";
    driver->state->build(model, driver->get_output_size());
    driver->model = model;
    return driver;
}
