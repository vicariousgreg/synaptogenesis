#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

#include "io/input.h"
#include "io/output.h"

void Driver::step_input() {
    // Clear all input
    this->state->clear_all_input();

    // Run input modules
    // If no module, clear the input
    for (int i = 0 ; i < this->model->input_drivers.size(); ++i)
        this->model->input_drivers[i]->feed_input(this->state);

    // Calculate inputs for connections
    for (int cid = 0 ; cid < this->model->num_connections; ++cid)
        step_connection(this->model->connections[cid]);
}

void Driver::print_output() {
    // Run output modules
    // If no module, skip layer
    for (int i = 0 ; i < this->model->output_drivers.size(); ++i)
        this->model->output_drivers[i]->report_output(this->state);
}

Driver* build_driver(Model* model) {
    Driver* driver;
    if (model->driver_string == "izhikevich")
        driver = new IzhikevichDriver();
    else if (model->driver_string == "rate_encoding")
        driver = new RateEncodingDriver();
    else
        throw "Unrecognized driver type!";
    driver->state->build(model);
    driver->model = model;
    return driver;
}
