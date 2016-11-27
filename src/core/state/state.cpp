#include <cstdlib>
#include <cstring>

#include "state/state.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

State::State(Model *model)
        : attributes(build_attributes(model)),
          weight_matrices(new WeightMatrices(model, 3)) {
    int input_output_size = attributes->get_num_neurons(INPUT_OUTPUT);
    int input_size = input_output_size + attributes->get_num_neurons(INPUT);
    int output_size = input_output_size + attributes->get_num_neurons(OUTPUT);

    this->buffer = new Buffer(
        input_output_size + attributes->get_num_neurons(INPUT),
        input_output_size + attributes->get_num_neurons(OUTPUT),
        attributes->get_output_type());

#ifdef PARALLEL
    // Create streams
    cudaStreamCreate(&this->io_stream);
    cudaStreamCreate(&this->state_stream);

    // Create events
    input_event = new cudaEvent_t;
    clear_event = new cudaEvent_t;
    output_calc_event = new cudaEvent_t;
    output_event = new cudaEvent_t;
#endif
}

State::~State() {
    delete this->weight_matrices;
    delete this->attributes;
    delete this->buffer;

#ifdef PARALLEL
    cudaStreamDestroy(io_stream);
    cudaStreamDestroy(state_stream);
    cudaEventDestroy(*input_event);
    cudaEventDestroy(*clear_event);
    cudaEventDestroy(*output_calc_event);
    cudaEventDestroy(*output_event);
    delete input_event;
    delete clear_event;
    delete output_calc_event;
    delete output_event;
#endif
}

void State::reset() {
#ifdef PARALLEL
    // Reset events
    unsigned int flags = cudaEventDisableTiming;
    unsigned int io_flags = flags & cudaEventBlockingSync;
    cudaEventCreateWithFlags(input_event, io_flags);
    cudaEventCreateWithFlags(clear_event, flags);
    cudaEventCreateWithFlags(output_calc_event, flags);
    cudaEventCreateWithFlags(output_event, io_flags);
#endif

    // Clear inputs that aren't connected to sensory input
    float *input = this->attributes->get_input();
    int offset = this->attributes->get_start_index(OUTPUT);
    int count = this->attributes->get_num_neurons() - offset;

#ifdef PARALLEL
    if (count > 0) {
        int threads = calc_threads(count);
        int blocks = calc_blocks(count);
        // Use the kernel stream
        clear_data<<<blocks, threads, 0, this->state_stream>>>(
            input + offset, count);
        cudaCheckError("Failed to clear inputs!");
    }
    cudaEventRecord(*this->clear_event, this->state_stream);
    cudaStreamWaitEvent(this->state_stream, *this->input_event, 0);
#else
    if (count > 0)
        clear_data(input + offset, count);
#endif
}

void State::update_states(int start_index, int count) {
#ifdef PARALLEL
    this->attributes->update(start_index, count, state_stream);
#else
    this->attributes->update(start_index, count);
#endif
}

void State::get_input() {
#ifdef PARALLEL
    this->attributes->get_input_from(buffer, io_stream);
    cudaEventRecord(*this->input_event, this->io_stream);
#else
    this->attributes->get_input_from(buffer);
#endif
}

void State::send_output() {
#ifdef PARALLEL
    // Make sure to wait for output calc event
    cudaStreamWaitEvent(this->io_stream, *this->output_calc_event, 0);
    this->attributes->send_output_to(buffer, io_stream);
    cudaEventRecord(*this->output_event, this->io_stream);
#else
    this->attributes->send_output_to(buffer);
#endif
}

void State::wait_for_input() {
#ifdef PARALLEL
    cudaEventSynchronize(*this->input_event);
#endif
}

void State::wait_for_output() {
#ifdef PARALLEL
    cudaEventSynchronize(*this->output_event);
#endif
}

void State::update_output_states() {
    update_states(INPUT_OUTPUT);
    update_states(OUTPUT);
#ifdef PARALLEL
    cudaEventRecord(*this->output_calc_event, this->state_stream);
#endif
}

void State::update_non_output_states() {
    update_states(INPUT);
    update_states(INTERNAL);
}

void State::update_all_states() {
    this->update_states(0, attributes->get_num_neurons());
}

void State::update_states(IOType layer_type) {
    int start_index = attributes->get_start_index(layer_type);
    int count = attributes->get_num_neurons(layer_type);
    if (count > 0)
        this->update_states(start_index, count);
}
