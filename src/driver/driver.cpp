#include "driver/driver.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"

Driver::Driver(Model *model, State *state)
        : state(state), stream_cluster(model, state) { }

Driver::~Driver() {
    delete this->state;
}

void Driver::stage_clear() {
    stream_cluster.reset();

#ifdef PARALLEL
    // Initialize state for timestep
    this->state->initialize();

    // Start input clearing
    this->state->clear_input();

    // Ensure all layer streams wait for appropriate input
    stream_cluster.wait_event(INPUT, this->state->input_event);
    stream_cluster.wait_event(INPUT_OUTPUT, this->state->input_event);
    stream_cluster.wait_event(OUTPUT, this->state->clear_event);
    stream_cluster.wait_event(INTERNAL, this->state->clear_event);

    // Launch output relevant computations
    stream_cluster.schedule_execution_from(INPUT_OUTPUT);
    stream_cluster.schedule_execution_from(OUTPUT);
    stream_cluster.schedule_execution_to(OUTPUT);
    stream_cluster.schedule_execution_to(INPUT_OUTPUT);
    stream_cluster.dispatch(this);

    // Wait for output computations to finish
    stream_cluster.block_stream_from(OUTPUT, this->state->state_stream);
    stream_cluster.block_stream_from(INPUT_OUTPUT, this->state->state_stream);

    stream_cluster.block_stream_to(INPUT_OUTPUT, this->state->state_stream);
    stream_cluster.block_stream_to(OUTPUT, this->state->state_stream);

    // Output state computation
    this->state->step_output_states();

    // Launch remaining calculations
    stream_cluster.schedule_execution_to(INPUT);
    stream_cluster.schedule_execution_to(INTERNAL);
    stream_cluster.dispatch(this);
    // Block kernel stream until they are done
    stream_cluster.block_stream_to(INPUT, this->state->state_stream);
    stream_cluster.block_stream_to(INTERNAL, this->state->state_stream);

    // Launch final state computations
    this->state->step_non_output_states();

    // Wait for input stream to return
    cudaEventSynchronize(*this->state->input_event);
#else
    this->state->clear_input();
#endif
}

void Driver::stage_input() {
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
    stream_cluster.schedule_execution_from(OUTPUT);
    stream_cluster.schedule_execution_from(INPUT_OUTPUT);
    stream_cluster.schedule_execution_to(OUTPUT);
    stream_cluster.schedule_execution_to(INPUT_OUTPUT);
    stream_cluster.dispatch(this);
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
    stream_cluster.schedule_execution_to(INPUT);
    stream_cluster.schedule_execution_to(INTERNAL);
    stream_cluster.dispatch(this);
    this->state->step_non_output_states();
#endif
}


void Driver::stage_weights() {
    stream_cluster.schedule_weight_update();
    stream_cluster.dispatch(this);
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
