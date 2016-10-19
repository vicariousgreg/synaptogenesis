#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

static void add_instructions(std::vector<Instruction *> &instructions,
        Layer *layer, State *state, int timesteps_per_output) {
    for (int i = 0; i < layer->input_connections.size(); ++i) {
        Connection *conn = layer->input_connections[i];
        int word_index = HISTORY_SIZE - 1 -
            (conn->delay / timesteps_per_output);
        if (word_index < 0) throw "Invalid delay in connection!";

        Output* out =
            &state->output[state->num_neurons * word_index];

        instructions.push_back(
            new Instruction(conn,
                out,
                state->input,
                state->get_matrix(conn->id)));
    }
}

void Driver::build_instructions(Model *model, int timesteps_per_output) {
    for (int i = 0; i < model->input_layers.size(); ++i)
        add_instructions(this->input_instructions,
            model->input_layers[i], this->state, timesteps_per_output);

    for (int i = 0; i < model->io_layers.size(); ++i)
        add_instructions(this->io_instructions,
            model->io_layers[i], this->state, timesteps_per_output);

    for (int i = 0; i < model->output_layers.size(); ++i)
        add_instructions(this->output_instructions,
            model->output_layers[i], this->state, timesteps_per_output);

    for (int i = 0; i < model->internal_layers.size(); ++i)
        add_instructions(this->internal_instructions,
            model->internal_layers[i], this->state, timesteps_per_output);
}

void Driver::step_input(Buffer *buffer) {
    this->state->get_input_from(buffer);
}

void Driver::step_connections(std::vector<Instruction*> &instructions) {
    for (int i = 0; i < instructions.size(); ++i)
        this->step_connection(instructions[i]);
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
