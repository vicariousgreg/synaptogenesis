#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

void Driver::step_input(Buffer *buffer) {
    this->state->get_input_from(buffer);
}

void Driver::step_connections() {
    // Calculate inputs for connections
    for (int cid = 0 ; cid < this->model->connections.size(); ++cid)
        step_connection(this->model->connections[cid]);
}

void Driver::step_output(Buffer *buffer) {
    this->state->send_output_to(buffer);
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
