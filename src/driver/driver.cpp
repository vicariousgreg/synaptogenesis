#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

void Driver::build_instructions(Model *model, int timesteps_per_output) {
    for (int layer_type = 0; layer_type < LAYER_TYPE_SIZE; ++layer_type) {
        for (int i = 0; i < model->layers[layer_type].size(); ++i) {
            Layer *layer = model->layers[layer_type][i];
            for (int i = 0; i < layer->input_connections.size(); ++i) {
                Connection *conn = layer->input_connections[i];
                int word_index = HISTORY_SIZE - 1 -
                    (conn->delay / timesteps_per_output);
                if (word_index < 0) throw "Invalid delay in connection!";

                Output* out =
                    &this->state->output[this->state->num_neurons * word_index];

                this->instructions[layer_type].push_back(
                    new Instruction(conn,
                        out,
                        this->state->input,
                        this->state->get_matrix(conn->id)));
            }
        }
    }
}

void Driver::step_input(Buffer *buffer) {
    this->state->get_input_from(buffer);
}

void Driver::step_connections(LayerType layer_type) {
    for (int i = 0; i < instructions[layer_type].size(); ++i)
        this->step_connection(instructions[layer_type][i]);
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
