#include "driver.h"
#include "../implementation/izhikevich_driver.h"
#include "../implementation/rate_encoding_driver.h"

#include "input.h"
#include "output.h"

void Driver::step_input() {
    // Run input modules
    // If no module, clear the input
    for (int i = 0 ; i < this->model->num_layers; ++i) {
        Input *input = this->model->layers[i]->input;
        if (input == NULL) {
            this->state->clear_input(i);
        } else {
            input->feed_input(this->state);
        }
    }

    // Calculate inputs for connections
    for (int cid = 0 ; cid < this->model->num_connections; ++cid) {
        Connection *conn = this->model->connections[cid];
        switch (conn->type) {
            case (FULLY_CONNECTED):
                step_connection_fully_connected(conn);
                break;
            case (ONE_TO_ONE):
                step_connection_one_to_one(conn);
                break;
            case (DIVERGENT):
                step_connection_divergent(conn);
                break;
            case (CONVERGENT):
                step_connection_convergent(conn, false);
                break;
            case (CONVOLUTIONAL):
                step_connection_convergent(conn, true);
                break;
            default:
                throw "Unimplemented connection type!";
        }
    }
}

void Driver::print_output() {
    // Run output modules
    // If no module, skip layer
    for (int i = 0 ; i < this->model->num_layers; ++i) {
        Output *output = this->model->layers[i]->output;
        if (output != NULL) {
            output->report_output(this->state);
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
