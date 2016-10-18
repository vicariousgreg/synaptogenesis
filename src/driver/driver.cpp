#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

void Driver::build_instructions(Model *model, int timesteps_per_output) {
    for (int i = 0; i < model->connections.size(); ++i) {
        Connection *conn = model->connections[i];
        int word_index = HISTORY_SIZE - 1 -
            (conn->delay / timesteps_per_output);
        if (word_index < 0) throw "Invalid delay in connection!";

        Output* out =
            &this->state->output[this->state->num_neurons * word_index];

        this->instructions.push_back(
            new Instruction(conn,
                out,
                this->state->input,
                this->state->get_matrix(conn->id)));
    }
}

void Driver::step_input(Buffer *buffer) {
    this->state->get_input_from(buffer);
}

void Driver::step_connections() {
    for (int i = 0; i < this->instructions.size(); ++i)
        this->step_connection(this->instructions[i]);
}

void Driver::step_output(Buffer *buffer) {
    this->state->send_output_to(buffer);
}

Driver* build_driver(Model* model) {
    Driver* driver;
    if (model->driver_string == "izhikevich")
        driver = new IzhikevichDriver(model);
    else if (model->driver_string == "rate_encoding")
        driver = new RateEncodingDriver(model);
    else
        throw "Unrecognized driver type!";
    driver->build_instructions(model, driver->get_timesteps_per_output());
    return driver;
}
