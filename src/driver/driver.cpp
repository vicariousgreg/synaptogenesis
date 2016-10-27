#include "driver/driver.h"

void Driver::stage_clear() {
    // Reset stream cluster and state for timestep
    stream_cluster.reset();
    this->state->reset();

#ifdef PARALLEL
    // If parallel, schedule everything now
    // Events will ensure processing waits until ready

    // Launch output relevant computations
    stream_cluster.schedule_output_calculations();
    stream_cluster.dispatch(this);

    // Wait for output computations to finish
    stream_cluster.block_state_on_output_calculations();

    // Output state computation
    this->state->update_output_states();

    // Launch remaining calculations
    stream_cluster.schedule_non_output_calculations();
    stream_cluster.dispatch(this);

    // Block kernel stream until they are done
    stream_cluster.block_state_on_non_output_calculations();

    // Launch final state computations
    this->state->update_non_output_states();
#endif
}

void Driver::stage_input() {
    // Start input streaming
    this->state->get_input();

#ifdef PARALLEL
    // Wait for it to finish
    this->state->wait_for_input();
#endif
}

void Driver::stage_calc_output() {
#ifdef PARALLEL
#else
    stream_cluster.schedule_output_calculations();
    stream_cluster.dispatch(this);
    this->state->update_output_states();
#endif
}

void Driver::stage_send_output(bool mock) {
    // Stream output
    this->state->send_output(mock);

#ifdef PARALLEL
    // Wait for it to finish
    if (not mock) this->state->wait_for_output();
#endif
}

void Driver::stage_remaining() {
#ifdef PARALLEL
    // Synchronize and check for errors
    cudaSync();
    cudaCheckError(NULL);
#else
    stream_cluster.schedule_non_output_calculations();
    stream_cluster.dispatch(this);
    this->state->update_non_output_states();
#endif
}

void Driver::stage_weights() {
    stream_cluster.schedule_weight_update();
    stream_cluster.dispatch(this);
}
