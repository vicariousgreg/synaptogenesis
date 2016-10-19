#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

void Driver::build_instructions(Model *model, int timesteps_per_output) {
    for (int layer_type = 0; layer_type < LAYER_TYPE_SIZE; ++layer_type) {
        for (int i = 0; i < model->layers[layer_type].size(); ++i) {
            Layer *layer = model->layers[layer_type][i];
            for (int j = 0; j < layer->input_connections.size(); ++j) {
                Connection *conn = layer->input_connections[j];
                int word_index = HISTORY_SIZE - 1 -
                    (conn->delay / timesteps_per_output);
                if (word_index < 0) throw "Invalid delay in connection!";

                Output* out =
                    &this->state->output[this->state->total_neurons * word_index];

                Instruction *inst =
                    new Instruction(conn,
                        out,
                        this->state->input,
                        this->state->get_matrix(conn->id));
                this->instructions[layer_type].push_back(inst);
                this->all_instructions.push_back(inst);
            }
        }
    }
}

void Driver::step_input(Buffer *buffer) {
    this->state->get_input_from(buffer);
    clear_input(this->state->input,
        this->state->start_index[OUTPUT],
        this->state->total_neurons);
}

void Driver::step_connections() {
    for (int i = 0; i < all_instructions.size(); ++i)
        this->update_connection(all_instructions[i]);
}

void Driver::step_connections(LayerType layer_type) {
    for (int i = 0; i < instructions[layer_type].size(); ++i)
        this->update_connection(instructions[layer_type][i]);
}

void Driver::step_state() {
    this->update_state(0, this->state->total_neurons);
}

void Driver::step_state(LayerType layer_type) {
    int start_index = this->state->start_index[layer_type];
    int count = this->state->num_neurons[layer_type];
    if (count > 0)
        this->update_state(start_index, count);
}

void Driver::step_output(Buffer *buffer) {
    this->state->send_output_to(buffer);
}

void Driver::step_weights() {
    for (int i = 0; i < all_instructions.size(); ++i)
        this->update_weights(all_instructions[i]);
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
