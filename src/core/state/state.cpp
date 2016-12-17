#include <cstdlib>
#include <cstring>

#include "state/state.h"
#include "state/izhikevich_state.h"
#include "state/rate_encoding_state.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

State *build_state(Model *model) {
    State *state;
    if (model->engine_name == "izhikevich")
        state = new IzhikevichState(model);
    else if (model->engine_name == "rate_encoding")
        state = new RateEncodingState(model);
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized engine type!");
    return state;
}

State::State(Model *model, OutputType output_type, int matrix_depth)
        : output_type(output_type),
          weight_matrices(new WeightMatrices(model, matrix_depth)) {
    // Get neuron counts
    this->total_neurons = model->num_neurons;

    // Determine start indices and number of neurons for each type
    int curr_index = 0;
    for (int layer_type = 0; layer_type < IO_TYPE_SIZE; ++layer_type) {
        std::vector<Layer*> layers = model->layers[layer_type];
        int size = 0;
        for (int i = 0; i < layers.size(); ++i)
            size += layers[i]->size;

        this->start_indices[layer_type] = curr_index;
        this->num_neurons[layer_type] = size;
        curr_index += size;
    }

    // Allocate space for input and output
    float* local_input = (float*) allocate_host(
        this->total_neurons, sizeof(float));
    Output* local_output = (Output*) allocate_host(
        this->total_neurons * HISTORY_SIZE, sizeof(Output));

    // Retrieve attribute kernel
    get_attribute_kernel(&this->attribute_kernel, model->engine_name);

#ifdef PARALLEL
    // Copy data to device, then free from host
    this->input = (float*)
        allocate_device(this->total_neurons, sizeof(float), local_input);
    this->output = (Output*)
        allocate_device(this->total_neurons * HISTORY_SIZE, sizeof(Output), local_output);
    free(local_input);
    free(local_output);

    // Set streams to NULL for now
    this->state_stream = NULL;
    this->io_stream = NULL;
#else
    // Simply set pointers
    this->input = local_input;
    this->output = local_output;
#endif
    // Create pointer to most recent word of output
    this->recent_output = this->output + ((HISTORY_SIZE-1) * this->total_neurons);

    ////////////////////////////////////////
    int input_output_size = this->num_neurons[INPUT_OUTPUT];
    int input_size = input_output_size + this->num_neurons[INPUT];
    int output_size = input_output_size + this->num_neurons[OUTPUT];
    this->buffer = new Buffer(input_size, output_size, output_type); 

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
    delete this->buffer;

#ifdef PARALLEL
    cudaFree(this->input);
    cudaFree(this->output);
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
#else
    free(this->input);
    free(this->output);
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

    // Clear inputs
    this->clear_input();
}

void State::clear_input() {
    // Clear inputs that aren't connected to sensory input
    int offset = this->start_indices[OUTPUT];
    int count = this->total_neurons - offset;

#ifdef PARALLEL
    if (count > 0) {
        int threads = calc_threads(count);
        int blocks = calc_blocks(count);
        // Use the kernel stream
        clear_data<<<blocks, threads, 0, this->state_stream>>>(
            this->input + offset, count);
        cudaCheckError("Failed to clear inputs!");
    }
    cudaEventRecord(*this->clear_event, this->state_stream);
    cudaStreamWaitEvent(this->state_stream, *this->input_event, 0);
#else
    if (count > 0) clear_data(this->input + offset, count);
#endif
}

void State::transfer_input() {
    int index = this->start_indices[INPUT];
    int count = this->num_neurons[INPUT] + this->num_neurons[INPUT_OUTPUT];
    if (count != 0) {
#ifdef PARALLEL
        // Copy to GPU from local location
        cudaMemcpyAsync(this->input + index, this->buffer->get_input(),
            count * sizeof(float), cudaMemcpyHostToDevice, this->io_stream);
#else
        memcpy(this->input + index, this->buffer->get_input(),
            count * sizeof(float));
#endif
    }

#ifdef PARALLEL
    cudaEventRecord(*this->input_event, this->io_stream);
#endif
}

void State::transfer_output() {
    int index = this->start_indices[INPUT_OUTPUT];
    int count = this->num_neurons[INPUT_OUTPUT] + this->num_neurons[OUTPUT];
    if (count != 0) {
#ifdef PARALLEL
        // Make sure to wait for output calc event
        cudaStreamWaitEvent(this->io_stream, *this->output_calc_event, 0);

        // Copy from GPU to local location
        cudaMemcpyAsync(buffer->get_output(), this->recent_output + index,
            count * sizeof(Output), cudaMemcpyDeviceToHost, this->io_stream);
#else
        memcpy(buffer->get_output(), this->recent_output + index,
            count * sizeof(Output));
#endif
    }

#ifdef PARALLEL
    cudaEventRecord(*this->output_event, this->io_stream);
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

    this->attribute_kernel<<<blocks, threads, 0, this->state_stream>>>(
        this->device_pointer, start_index, count, total_neurons);
    cudaCheckError("Failed to update neuron state/output!");
#else
    this->attribute_kernel(
        this, start_index, count, total_neurons);
#endif
}

void State::update_all_states() {
    this->update_states(0, this->total_neurons);
}

void State::update_states(IOType layer_type) {
    int start_index = this->start_indices[layer_type];
    int count = num_neurons[layer_type];
    if (count > 0)
        this->update_states(start_index, count);
}
