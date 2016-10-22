#include "driver/driver.h"
#include "driver/izhikevich_driver.h"
#include "driver/rate_encoding_driver.h"

Driver::Driver() {
#ifdef PARALLEL
    cudaStreamCreate(&this->io_stream);
    cudaStreamCreate(&this->kernel_stream);
    this->curr_stream = &this->kernel_stream;
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

        // Stream cluster
        stream_clusters[to_layer->type].add_instruction(to_layer, inst, from_layer->type);
    }
}

#ifdef PARALLEL
void Driver::wait_event(IOType to_type, cudaEvent_t event) {
    stream_clusters[to_type].wait_event(event);
}
#endif

void Driver::launch_from(IOType from_type) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        stream_clusters[i].execute(this, from_type);
}

void Driver::launch_to(IOType to_type) {
    stream_clusters[to_type].execute(this);
}

void Driver::launch(IOType from_type, IOType to_type) {
    stream_clusters[to_type].execute(this, from_type);
}

///////
void Driver::stage_input(Buffer *buffer) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        stream_clusters[i].reset();

#ifdef PARALLEL
    // Create events
    unsigned int flags = cudaEventDisableTiming;
    unsigned int io_flags = flags & cudaEventBlockingSync;
    cudaEventCreateWithFlags(&input_event, io_flags);
    cudaEventCreateWithFlags(&clear_event, flags);
    cudaEventCreateWithFlags(&output_calc_event, flags);
    cudaEventCreateWithFlags(&output_event, io_flags);

    // Start input clearing
    this->curr_stream = &this->kernel_stream;
    clear_input(this->state->input,
        this->state->start_index[OUTPUT],
        this->state->total_neurons);
    cudaEventRecord(this->clear_event, *this->curr_stream);

    // Start input streaming
    this->state->get_input_from(buffer, this->io_stream);
    cudaEventRecord(this->input_event, this->io_stream);

    // Ensure all layer streams wait for appropriate input
    wait_event(INPUT, this->input_event);
    wait_event(INPUT_OUTPUT, this->input_event);
    wait_event(OUTPUT, this->clear_event);
    wait_event(INTERNAL, this->clear_event);

    // Launch output relevant computations
    launch_from(OUTPUT);
    launch_from(INPUT_OUTPUT);
    launch_to(OUTPUT);
    launch_to(INPUT_OUTPUT);

    // Wait for output computations to finish
    for (int i = 0; i < IO_TYPE_SIZE; ++i) {
        stream_clusters[i].block_stream(this->kernel_stream, OUTPUT);
        stream_clusters[i].block_stream(this->kernel_stream, INPUT_OUTPUT);
    }
    stream_clusters[INPUT_OUTPUT].block_stream(this->kernel_stream);
    stream_clusters[OUTPUT].block_stream(this->kernel_stream);

    // Output state computation
    this->curr_stream = &this->kernel_stream;
    step_states(INPUT_OUTPUT);
    step_states(OUTPUT);
    cudaEventRecord(this->output_calc_event, *this->curr_stream);

    // Launch remaining calculations
    launch_to(INPUT);
    launch_to(INTERNAL);
    // Block kernel stream until they are done
    stream_clusters[INPUT].block_stream(this->kernel_stream);
    stream_clusters[INTERNAL].block_stream(this->kernel_stream);

    // Launch final state computations
    this->curr_stream = &this->kernel_stream;
    step_states(INPUT);
    step_states(INTERNAL);

    // Wait for input stream to return
    cudaEventSynchronize(this->input_event);
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
    launch_from(OUTPUT);
    launch_from(INPUT_OUTPUT);
    launch_to(OUTPUT);
    launch_to(INPUT_OUTPUT);
    step_states(INPUT_OUTPUT);
    step_states(OUTPUT);
#endif
}

void Driver::stage_send_output(Buffer *buffer) {
#ifdef PARALLEL
    // Wait for output calculation to complete
    cudaEventSynchronize(this->output_calc_event);

    // Start stream, and wait for it to return
    this->state->send_output_to(buffer, this->io_stream);
    cudaEventSynchronize(this->output_event);
#else
    this->state->send_output_to(buffer);
#endif
}

void Driver::stage_remaining() {
#ifdef PARALLEL
    // Synchronize, then delete events
    cudaSync();
    cudaCheckError(NULL);
#else
    launch_to(INPUT);
    launch_to(INTERNAL);
    step_states(INPUT);
    step_states(INTERNAL);
    step_weights();
#endif
}


///////

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
