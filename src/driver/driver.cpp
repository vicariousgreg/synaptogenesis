#include "driver/driver.h"
#include "driver/scheduler.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"

Driver::Driver(Model *model, State *state) : state(state) {
    // Build instructions
    for (int i = 0; i < model->connections.size(); ++i) {
        Connection *conn = model->connections[i];

        // Create instruction
        Instruction *inst = new Instruction(conn, this->state);
        this->all_instructions.push_back(inst);

        // Stream cluster
        stream_clusters[conn->to_layer->type].add_instruction(
            conn->to_layer, inst, conn->from_layer->type);
    }
}

Driver::~Driver() {
    delete this->state;
    for (int i = 0; i < this->all_instructions.size(); ++i)
        delete this->all_instructions[i];
}

#ifdef PARALLEL
void Driver::wait_event(IOType to_type, cudaEvent_t *event) {
    stream_clusters[to_type].wait_event(event);
}
#endif

void Driver::schedule_from(IOType from_type) {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        stream_clusters[i].schedule_execution(from_type);
}

void Driver::schedule_to(IOType to_type) {
    stream_clusters[to_type].schedule_execution();
}

void Driver::stage_clear() {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        stream_clusters[i].reset();

#ifdef PARALLEL
    // Initialize state for timestep
    this->state->initialize();

    // Start input clearing
    this->state->clear_input();

    // Ensure all layer streams wait for appropriate input
    wait_event(INPUT, this->state->input_event);
    wait_event(INPUT_OUTPUT, this->state->input_event);
    wait_event(OUTPUT, this->state->clear_event);
    wait_event(INTERNAL, this->state->clear_event);

    // Launch output relevant computations
    schedule_from(INPUT_OUTPUT);
    schedule_from(OUTPUT);
    schedule_to(OUTPUT);
    schedule_to(INPUT_OUTPUT);
    Scheduler::get_instance()->dispatch(this);

    // Wait for output computations to finish
    for (int i = 0; i < IO_TYPE_SIZE; ++i) {
        stream_clusters[i].block_stream(this->state->state_stream, OUTPUT);
        stream_clusters[i].block_stream(this->state->state_stream, INPUT_OUTPUT);
    }
    stream_clusters[INPUT_OUTPUT].block_stream(this->state->state_stream);
    stream_clusters[OUTPUT].block_stream(this->state->state_stream);

    // Output state computation
    this->state->step_output_states();

    // Launch remaining calculations
    schedule_to(INPUT);
    schedule_to(INTERNAL);
    Scheduler::get_instance()->dispatch(this);
    // Block kernel stream until they are done
    stream_clusters[INPUT].block_stream(this->state->state_stream);
    stream_clusters[INTERNAL].block_stream(this->state->state_stream);

    // Launch final state computations
    this->state->step_non_output_states();

    // Wait for input stream to return
    cudaEventSynchronize(*this->state->input_event);
#else
    this->state->clear_input();
#endif
}

void Driver::stage_input() {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        stream_clusters[i].reset();

#ifdef PARALLEL
    // Start input streaming
    this->state->get_input();

    // Wait for input stream to return
    cudaEventSynchronize(*this->state->input_event);
#else
    this->state->get_input();
#endif
}

void Driver::stage_calc_output() {
#ifdef PARALLEL
#else
    schedule_from(OUTPUT);
    schedule_from(INPUT_OUTPUT);
    schedule_to(OUTPUT);
    schedule_to(INPUT_OUTPUT);
    Scheduler::get_instance()->dispatch(this);
    this->state->step_output_states();
#endif
}

void Driver::stage_send_output() {
#ifdef PARALLEL
    // Wait for output calculation to complete
    cudaEventSynchronize(*this->state->output_calc_event);

    // Start stream, and wait for it to return
    this->state->send_output();
    cudaEventSynchronize(*this->state->output_event);
#else
    this->state->send_output();
#endif
}

void Driver::stage_remaining() {
#ifdef PARALLEL
    // Synchronize and check for errors
    cudaSync();
    cudaCheckError(NULL);
#else
    schedule_to(INPUT);
    schedule_to(INTERNAL);
    Scheduler::get_instance()->dispatch(this);
    this->state->step_non_output_states();
    step_weights();
#endif
}


void Driver::step_weights() {
    for (int i = 0; i < IO_TYPE_SIZE; ++i)
        stream_clusters[i].schedule_weight_update();
    Scheduler::get_instance()->dispatch(this);
}

Driver* build_driver(Model* model) {
    State* state;
    if (model->driver_string == "izhikevich")
        state = new State(model, new IzhikevichAttributes(model), 1);
    else if (model->driver_string == "rate_encoding")
        state = new State(model, new RateEncodingAttributes(model), 1);
    else
        throw "Unrecognized driver type!";
    return new Driver(model, state);
}
