#include "driver.h"
#include "izhikevich_driver.h"
#include "rate_encoding_driver.h"

#include "input.h"

void Driver::step_input() {
    // Run input modules
    // If no module, clear the input
    for (int i = 0 ; i < this->model->num_layers; ++i) {
        Input *input = this->model->layers[i].input;
        if (input == NULL) {
            this->state->clear_input(i);
        } else {
            input->feed_input(this->state);
        }
    }

    // Calculate inputs for connections
    for (int cid = 0 ; cid < this->model->num_connections; ++cid) {
        Connection &conn = this->model->connections[cid];
        if (conn.type == FULLY_CONNECTED) {
            step_connection_fully_connected(conn);
        } else if (conn.type == ONE_TO_ONE) {
            step_connection_one_to_one(conn);
        }
    }
}

Driver* build_driver(Model* model) {
    Driver* driver;
    if (model->driver_string == "izhikevich") {
        driver = new IzhikevichDriver();
    } else if (model->driver_string == "rate_encoding") {
        driver = new RateEncodingDriver();
    } else {
        throw "Unrecognized driver type!";
    }
    driver->state->build(model);
    driver->model = model;
    return driver;
}
