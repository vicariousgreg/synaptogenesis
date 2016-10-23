#include <cstdlib>
#include <cstring>
#include <sstream>

#include "state/attributes.h"
#include "tools.h"
#include "parallel.h"

Attributes::Attributes(Model *model, OutputType output_type) :
        output_type(output_type) {
    // Get neuron counts
    this->total_neurons = model->num_neurons;

    // Determine start indices and number of neurons for each type
    int curr_index = 0;
    for (int layer_type = 0; layer_type < IO_TYPE_SIZE; ++layer_type) {
        std::vector<Layer*> layers = model->layers[layer_type];
        int size = 0;
        for (int i = 0; i < layers.size(); ++i)
            size += layers[i]->size;

        this->start_index[layer_type] = curr_index;
        this->num_neurons[layer_type] = size;
        curr_index += size;
    }

    // Allocate space for input and output
    float* local_input = (float*) allocate_host(
        this->total_neurons, sizeof(float));
    Output* local_output = (Output*) allocate_host(
        this->total_neurons * HISTORY_SIZE, sizeof(Output));

#ifdef PARALLEL
    // Copy data to device, then free from host
    this->input = (float*)
        allocate_device(this->total_neurons, sizeof(float), local_input);
    this->output = (Output*)
        allocate_device(this->total_neurons * HISTORY_SIZE, sizeof(Output), local_output);
    free(local_input);
    free(local_output);
#else
    // Simply set pointers
    this->input = local_input;
    this->output = local_output;
#endif
    // Create pointer to most recent word of output
    this->recent_output = this->output + ((HISTORY_SIZE-1) * this->total_neurons);
}

Attributes::~Attributes() {
#ifdef PARALLEL
    cudaFree(this->input);
    cudaFree(this->output);
#else
    free(this->input);
    free(this->output);
#endif
}

#ifdef PARALLEL
void Attributes::get_input_from(Buffer *buffer, cudaStream_t &stream) {
    int index = this->start_index[INPUT];
    int count = this->num_neurons[INPUT] + this->num_neurons[INPUT_OUTPUT];
    if (count != 0) {
        // Copy to GPU from local location
        cudaMemcpyAsync(this->input + index, buffer->get_input(),
            count * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
}
void Attributes::send_output_to(Buffer *buffer, cudaStream_t &stream) {
    int index = this->start_index[INPUT_OUTPUT];
    int count = this->num_neurons[INPUT_OUTPUT] + this->num_neurons[OUTPUT];
    if (count != 0) {
        // Copy from GPU to local location
        cudaMemcpyAsync(buffer->get_output(), this->recent_output + index,
            count * sizeof(Output), cudaMemcpyDeviceToHost, stream);
    }
}

#else
void Attributes::get_input_from(Buffer *buffer) {
    int index = this->start_index[INPUT];
    int count = this->num_neurons[INPUT] + this->num_neurons[INPUT_OUTPUT];
    if (count != 0) {
        memcpy(this->input + index, buffer->get_input(),
            count * sizeof(float));
    }
}

void Attributes::send_output_to(Buffer *buffer) {
    int index = this->start_index[INPUT_OUTPUT];
    int count = this->num_neurons[INPUT_OUTPUT] + this->num_neurons[OUTPUT];
    if (count != 0) {
        memcpy(buffer->get_output(), this->recent_output + index,
            count * sizeof(Output));
    }
}
#endif
