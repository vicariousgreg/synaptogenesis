#include <cstring>

#include "state/state.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

State::State(Model *model)
        : attributes(build_attributes(model)) {
    /* Set up weight matrices */
    for (auto & conn : model->get_connections()) {
        WeightMatrix* matrix = new WeightMatrix(conn, attributes->get_matrix_depth(conn));
        this->weight_matrices[conn] = matrix;
        this->attributes->process_weight_matrix(matrix);
#ifdef PARALLEL
        matrix->send_to_device();
#endif
    }

    // Create the buffer
    int input_output_size = attributes->get_num_neurons(INPUT_OUTPUT);
    int input_size = input_output_size + attributes->get_num_neurons(INPUT);
    int output_size = input_output_size + attributes->get_num_neurons(OUTPUT);
    this->buffer = new Buffer(input_size, output_size, attributes->output_type); 

#ifdef PARALLEL
    // Create streams
    cudaStreamCreate(&this->input_stream);
    cudaStreamCreate(&this->output_stream);
    cudaStreamCreate(&this->state_stream);

    // Create events
    input_event = new cudaEvent_t;
    clear_event = new cudaEvent_t;
    output_event = new cudaEvent_t;
    state_event = new cudaEvent_t;

    unsigned int flags = cudaEventDisableTiming;
    unsigned int io_flags = flags & cudaEventBlockingSync;
    cudaEventCreateWithFlags(input_event, io_flags);
    cudaEventCreateWithFlags(clear_event, flags);
    cudaEventCreateWithFlags(output_event, io_flags);
    cudaEventCreateWithFlags(state_event, io_flags);
#endif

}

State::~State() {
    delete attributes;
    delete buffer;
    for (auto matrix : this->weight_matrices) delete matrix.second;

#ifdef PARALLEL
    cudaStreamDestroy(input_stream);
    cudaStreamDestroy(output_stream);
    cudaStreamDestroy(state_stream);
    cudaEventDestroy(*input_event);
    cudaEventDestroy(*clear_event);
    cudaEventDestroy(*output_event);
    cudaEventDestroy(*state_event);
    delete input_event;
    delete clear_event;
    delete output_event;
    delete state_event;
#endif
}

void State::reset() {
    // Clear inputs that aren't connected to sensory input
    int offset = attributes->get_start_index(OUTPUT);
    int count = attributes->total_neurons - offset;

#ifdef PARALLEL
    if (count > 0) {
        int threads = calc_threads(count);
        int blocks = calc_blocks(count);

        // Use the state stream
        clear_data<<<blocks, threads, 0, this->state_stream>>>(
            attributes->input + offset, count);
        cudaCheckError("Failed to clear inputs!");
    }
    cudaEventRecord(*this->clear_event, this->state_stream);
#else
    if (count > 0) clear_data(attributes->input + offset, count);
#endif
}

void State::transfer_input() {
    int index = attributes->get_start_index(INPUT);
    int count = attributes->get_num_neurons(INPUT) + attributes->get_num_neurons(INPUT_OUTPUT);
    if (count != 0) {
#ifdef PARALLEL
        // Copy to GPU from local location
        cudaMemcpyAsync(attributes->input + index, this->buffer->get_input(),
            count * sizeof(float), cudaMemcpyHostToDevice, this->input_stream);
    }
    cudaEventRecord(*this->input_event, this->input_stream);
#else
        memcpy(attributes->input + index, this->buffer->get_input(),
            count * sizeof(float));
    }
#endif
}

void State::transfer_output() {
    int index = attributes->get_start_index(INPUT_OUTPUT);
    int count = attributes->get_num_neurons(INPUT_OUTPUT)
              + attributes->get_num_neurons(OUTPUT);
    if (count != 0) {
#ifdef PARALLEL
        // Copy from GPU to local location
        cudaMemcpyAsync(buffer->get_output(), attributes->recent_output + index,
            count * sizeof(Output), cudaMemcpyDeviceToHost, this->output_stream);
    }
    cudaEventRecord(*this->output_event, this->output_stream);
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

void State::update_states(int start_index, int count) {
#ifdef PARALLEL
    int threads = calc_threads(count);
    int blocks = calc_blocks(count);

    attributes->get_attribute_kernel()<<<blocks, threads, 0, this->state_stream>>>(
        attributes->pointer, start_index, count);
    cudaCheckError("Failed to update neuron state/output!");
#else
    attributes->get_attribute_kernel()(attributes, start_index, count);
#endif
}

void State::update_states() {
    this->update_states(0, attributes->total_neurons);
#ifdef PARALLEL
    cudaEventRecord(*this->state_event, this->state_stream);
#endif
}

void State::update_states(Layer *layer) {
    this->update_states( layer->get_start_index(), layer->size);
}

void State::update_states(IOType layer_type) {
    int start_index = attributes->get_start_index(layer_type);
    int count = attributes->get_num_neurons(layer_type);
    if (count > 0) this->update_states(start_index, count);
}
