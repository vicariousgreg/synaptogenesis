#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

Driver::Driver() {
#ifdef PARALLEL
    cudaStreamCreate(&this->io_stream);

    for (int i = 0; i < NUM_KERNEL_STREAMS; ++i)
        cudaStreamCreate(&this->kernel_streams[i]);
    this->curr_stream = &this->kernel_streams[0];

    cudaEventCreate(&input_event);
    cudaEventCreate(&clear_event);
    cudaEventCreate(&io_event);
    cudaEventCreate(&xo_event);
    cudaEventCreate(&output_event);
#endif
}

void Driver::build_instructions(Model *model, int timesteps_per_output) {
    for (int i = 0; i < model->connections.size(); ++i) {
        Connection *conn = model->connections[i];
        Layer *from_layer = conn->from_layer;
        Layer *to_layer = conn->to_layer;

        // Set up word index
        int word_index = HISTORY_SIZE - 1 -
            (conn->delay / timesteps_per_output);
        if (word_index < 0) throw "Invalid delay in connection!";

        // Create instruction
        Output* out = this->state->output + 
            (this->state->total_neurons * word_index);
        Instruction *inst =
            new Instruction(conn,
                out,
                this->state->input,
                this->state->get_matrix(conn->id));
        this->all_instructions.push_back(inst);

        // Determine which category it falls in
        if (to_layer->type == INPUT) {
            if (from_layer->type == INPUT or from_layer->type == INTERNAL)
                this->instructions_i.push_back(inst);
            else
                this->instructions_io.push_back(inst);
        } else if (to_layer->type == INPUT_OUTPUT) {
            this->instructions_io.push_back(inst);
        } else if (to_layer->type == OUTPUT) {
            this->instructions_xo.push_back(inst);
        } else if (to_layer->type == INTERNAL) {
            if (from_layer->type == INPUT or from_layer->type == INTERNAL)
                this->instructions_x.push_back(inst);
            else
                this->instructions_xo.push_back(inst);
        }
    }
}

///////
void Driver::stage_input(Buffer *buffer) {
#ifdef PARALLEL
#else
    this->state->get_input_from(buffer);
#endif
}

void Driver::stage_calc_output() {
#ifdef PARALLEL
#else
    clear_input(this->state->input,
        this->state->start_index[OUTPUT],
        this->state->total_neurons);
    step_connections_io();
    step_connections_xo();
    step_states(INPUT_OUTPUT);
    step_states(OUTPUT);
#endif
}

void Driver::stage_send_output(Buffer *buffer) {
#ifdef PARALLEL
#else
    this->state->send_output_to(buffer);
#endif
}

void Driver::stage_remaining() {
#ifdef PARALLEL
#else
    step_connections_i();
    step_connections_x();
    step_states(INPUT);
    step_states(INTERNAL);
    step_weights();
#endif
}


///////

void Driver::step_connections(std::vector<Instruction* > instructions) {
    for (int i = 0; i < instructions.size(); ++i)
        this->update_connection(instructions[i]);
}

void Driver::step_all_connections() { step_connections(this->all_instructions); }
void Driver::step_connections_i() { step_connections(this->instructions_i); }
void Driver::step_connections_io() { step_connections(this->instructions_io); }
void Driver::step_connections_xo() { step_connections(this->instructions_xo); }
void Driver::step_connections_x() { step_connections(this->instructions_x); }


void Driver::step_all_states() {
    this->update_state(0, this->state->total_neurons);
}

void Driver::step_states(IOType layer_type) {
    int start_index = this->state->start_index[layer_type];
    int count = this->state->num_neurons[layer_type];
    if (count > 0)
        this->update_state(start_index, count);
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
