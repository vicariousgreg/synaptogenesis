#include <cstdlib>
#include <cstring>
#include <sstream>

#include "state/state.h"
#include "driver/kernel.h"
#include "tools.h"
#include "parallel.h"

State::State(Model *model, Attributes *attributes, int weight_depth)
        : attributes(attributes),
          weight_matrices(new WeightMatrices(model, weight_depth)) {
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

    this->initialize();
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

#ifdef PARALLEL
void State::initialize() {
    // Create events
    unsigned int flags = cudaEventDisableTiming;
    unsigned int io_flags = flags & cudaEventBlockingSync;
    cudaEventCreateWithFlags(input_event, io_flags);
    cudaEventCreateWithFlags(clear_event, flags);
    cudaEventCreateWithFlags(output_calc_event, flags);
    cudaEventCreateWithFlags(output_event, io_flags);
}
#endif

void State::update(int start_index, int count) {
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
    this->attributes->send_output_to(buffer, io_stream);
    cudaEventRecord(*this->output_event, this->io_stream);
#else
    this->attributes->send_output_to(buffer);
#endif
}

void State::clear_input() {
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
#else
    if (count > 0)
        clear_data(input + offset, count);
#endif
}

void State::step_output_states() {
    step_state(INPUT_OUTPUT);
    step_state(OUTPUT);
#ifdef PARALLEL
    cudaEventRecord(*this->output_calc_event, this->state_stream);
#endif
}

void State::step_non_output_states() {
    step_state(INPUT);
    step_state(INTERNAL);
}

void State::step_all_states() {
    this->update(0, attributes->get_num_neurons());
}

void State::step_state(IOType layer_type) {
    int start_index = attributes->get_start_index(layer_type);
    int count = attributes->get_num_neurons(layer_type);
    if (count > 0)
        this->update(start_index, count);
}
