#include <cstring>

#include "state/state.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

State::State(Model *model)
        : attributes(build_attributes(model)),
          weight_matrices(new WeightMatrices(model, attributes->get_matrix_depth())) {
    // Create the buffer
    int input_output_size = attributes->num_neurons[INPUT_OUTPUT];
    int input_size = input_output_size + attributes->num_neurons[INPUT];
    int output_size = input_output_size + attributes->num_neurons[OUTPUT];
    this->buffer = new Buffer(input_size, output_size, attributes->output_type); 

#ifdef PARALLEL
    // Create streams
    cudaStreamCreate(&this->io_stream);
    cudaStreamCreate(&this->state_stream);

    // Create events
    input_event = new cudaEvent_t;
    clear_event = new cudaEvent_t;
    output_calc_event = new cudaEvent_t;
    output_event = new cudaEvent_t;

    unsigned int flags = cudaEventDisableTiming;
    unsigned int io_flags = flags & cudaEventBlockingSync;
    cudaEventCreateWithFlags(input_event, io_flags);
    cudaEventCreateWithFlags(clear_event, flags);
    cudaEventCreateWithFlags(output_calc_event, flags);
    cudaEventCreateWithFlags(output_event, io_flags);
#endif

}

State::~State() {
    delete attributes;
    delete weight_matrices;
    delete buffer;

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
    // Clear inputs that aren't connected to sensory input
    int offset = attributes->start_indices[OUTPUT];
    int count = attributes->total_neurons - offset;

#ifdef PARALLEL
    if (count > 0) {
        int threads = calc_threads(count);
        int blocks = calc_blocks(count);
        // Use the kernel stream
        clear_data<<<blocks, threads, 0, this->state_stream>>>(
            attributes->input + offset, count);
        cudaCheckError("Failed to clear inputs!");
    }
    cudaEventRecord(*this->clear_event, this->state_stream);
    cudaStreamWaitEvent(this->state_stream, *this->input_event, 0);
#else
    if (count > 0) clear_data(attributes->input + offset, count);
#endif
}

void State::transfer_input() {
    int index = attributes->start_indices[INPUT];
    int count = attributes->num_neurons[INPUT] + attributes->num_neurons[INPUT_OUTPUT];
    if (count != 0) {
#ifdef PARALLEL
        // Copy to GPU from local location
        cudaMemcpyAsync(attributes->input + index, this->buffer->get_input(),
            count * sizeof(float), cudaMemcpyHostToDevice, this->io_stream);
    }
    cudaEventRecord(*this->input_event, this->io_stream);
#else
        memcpy(attributes->input + index, this->buffer->get_input(),
            count * sizeof(float));
    }
#endif
}

void State::transfer_output() {
    int index = attributes->start_indices[INPUT_OUTPUT];
    int count = attributes->num_neurons[INPUT_OUTPUT] + attributes->num_neurons[OUTPUT];
    if (count != 0) {
#ifdef PARALLEL
        // Make sure to wait for output calc event
        cudaStreamWaitEvent(this->io_stream, *this->output_calc_event, 0);

        // Copy from GPU to local location
        cudaMemcpyAsync(buffer->get_output(), attributes->recent_output + index,
            count * sizeof(Output), cudaMemcpyDeviceToHost, this->io_stream);
    }
    cudaEventRecord(*this->output_event, this->io_stream);
#else
        memcpy(buffer->get_output(), attributes->recent_output + index,
            count * sizeof(Output));
    }
#endif

}

#ifdef PARALLEL
void State::wait_for_input() {
    cudaEventSynchronize(*this->input_event);
}
void State::wait_for_output() {
    cudaEventSynchronize(*this->output_event);
}
#endif

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

void State::update_states(int start_index, int count) {
#ifdef PARALLEL
    int threads = calc_threads(count);
    int blocks = calc_blocks(count);

    attributes->attribute_kernel<<<blocks, threads, 0, this->state_stream>>>(
        attributes->device_pointer, start_index, count);
    cudaCheckError("Failed to update neuron state/output!");
#else
    attributes->attribute_kernel(attributes, start_index, count);
#endif
}

void State::update_all_states() {
    this->update_states(0, attributes->total_neurons);
}

void State::update_states(IOType layer_type) {
    int start_index = attributes->start_indices[layer_type];
    int count = attributes->num_neurons[layer_type];
    if (count > 0)
        this->update_states(start_index, count);
}
