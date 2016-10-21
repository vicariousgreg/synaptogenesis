#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

Driver::Driver() {
#ifdef PARALLEL
    cudaStreamCreate(&this->io_stream);

    for (int i = 0; i < NUM_KERNEL_STREAMS; ++i)
        cudaStreamCreate(&this->kernel_streams[i]);
    this->curr_stream = &this->kernel_streams[0];
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
        if (from_layer->type == INPUT_OUTPUT or
            from_layer->type == OUTPUT) {
            // B instructions depend on IO or O output and environment input
            if (to_layer->type == INPUT or
                to_layer->type == INPUT_OUTPUT)
                this->instructions_b.push_back(inst);
            // B instructions depend on IO or O output only
            else
                this->instructions_a.push_back(inst);
        // C instructions calculate for IO/O connections but don't fall into A or B
        } else if (to_layer->type == INPUT_OUTPUT or to_layer->type == OUTPUT) {
            this->instructions_c.push_back(inst);
        // D instructions are everything else
        } else {
            this->instructions_d.push_back(inst);
        }
    }
}

///////

void Driver::stage_one(Buffer *buffer){
#ifdef PARALLEL
    // Ensure full synchronization
    cudaStreamSynchronize(this->kernel_streams[0]);
    cudaStreamSynchronize(this->kernel_streams[1]);

    // Launch input clearing on kernel stream 0
    this->curr_stream = &this->kernel_streams[0];
    clear_input(this->state->input,
        this->state->start_index[OUTPUT],
        this->state->total_neurons);
    step_connections_a();

    // Launch input copy on io stream
    this->state->get_input_from(buffer, this->io_stream);

    // Wait for input copy to finish
    cudaStreamSynchronize(this->io_stream);
#else
    this->state->get_input_from(buffer);
    clear_input(this->state->input,
        this->state->start_index[OUTPUT],
        this->state->total_neurons);
#endif
}

void Driver::stage_two(){
#ifdef PARALLEL
    // Wait for input clearing to finish
    cudaStreamSynchronize(this->kernel_streams[0]);

    // Launch connection computations for a, b, and c on separate streams
    this->curr_stream = &this->kernel_streams[1];
    step_connections_b();
    this->curr_stream = &this->kernel_streams[2];
    step_connections_c();
#else
    step_connections_a();
    step_connections_b();
    step_connections_c();
#endif
}

void Driver::stage_three() {
#ifdef PARALLEL
    // Wait for stage two to finish
    cudaStreamSynchronize(this->kernel_streams[0]);
    cudaStreamSynchronize(this->kernel_streams[1]);
    cudaStreamSynchronize(this->kernel_streams[2]);

    // Compute state/output for IO and O layers on separate streams
    this->curr_stream = &this->kernel_streams[0];
    step_states(INPUT_OUTPUT);

    this->curr_stream = &this->kernel_streams[1];
    step_states(OUTPUT);
#else
    step_states(INPUT_OUTPUT);
    step_states(OUTPUT);
#endif
}

void Driver::stage_four(Buffer *buffer){
#ifdef PARALLEL
    // Wait for computation to finish
    cudaStreamSynchronize(this->kernel_streams[0]);
    cudaStreamSynchronize(this->kernel_streams[1]);

    // Launch output copy
    this->state->send_output_to(buffer, this->io_stream);

    // Wait for output copy to return
    cudaStreamSynchronize(this->io_stream);
#else
    this->state->send_output_to(buffer);
#endif
}

void Driver::stage_five(){
#ifdef PARALLEL
    // Launch D instructions
    this->curr_stream = &this->kernel_streams[0];
    step_connections_d();

    // Wait for D computation
    cudaStreamSynchronize(this->kernel_streams[0]);

    // Compute state/output for I and In layers on separate streams
    this->curr_stream = &this->kernel_streams[0];
    step_states(INPUT);

    this->curr_stream = &this->kernel_streams[1];
    step_states(INTERNAL);
#else
    step_connections_d();
    step_states(INPUT);
    step_states(INTERNAL);
#endif
}

void Driver::step_weights() {
    for (int i = 0; i < all_instructions.size(); ++i)
        this->update_weights(all_instructions[i]);
}


///////

void Driver::step_connections(std::vector<Instruction* > instructions) {
    for (int i = 0; i < instructions.size(); ++i)
        this->update_connection(instructions[i]);
}

void Driver::step_all_connections() { step_connections(this->all_instructions); }
void Driver::step_connections_a() { step_connections(this->instructions_a); }
void Driver::step_connections_b() { step_connections(this->instructions_b); }
void Driver::step_connections_c() { step_connections(this->instructions_c); }
void Driver::step_connections_d() { step_connections(this->instructions_d); }


void Driver::step_all_states() {
    this->update_state(0, this->state->total_neurons);
}

void Driver::step_states(IOType layer_type) {
    int start_index = this->state->start_index[layer_type];
    int count = this->state->num_neurons[layer_type];
    if (count > 0)
        this->update_state(start_index, count);
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
