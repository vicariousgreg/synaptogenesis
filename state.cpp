#include <cstdlib>
#include <cstdio>

#include "state.h"
#include "model.h"
#include "tools.h"
#include "constants.h"
#include "parallel.h"

void State::build(Model model) {
    int num_neurons = model.num_neurons;
    this->num_neurons = num_neurons;

#ifdef PARALLEL
    // Local spikes for output reporting
    this->local_spikes = (int*)calloc(num_neurons, sizeof(int));
    this->local_current = (float*)calloc(num_neurons, sizeof(float));
    if (this->local_spikes == NULL or this->local_current == NULL)
        throw "Failed to allocate space on host for local copies of neuron state!";

    // Allocate space on GPU
    cudaMalloc(&this->current, num_neurons * sizeof(float));
    cudaMalloc(&this->voltage, num_neurons * sizeof(float));
    cudaMalloc(&this->recovery, num_neurons * sizeof(float));
    cudaMalloc(&this->spikes, HISTORY_SIZE * num_neurons * sizeof(int));
    cudaMalloc(&this->neuron_parameters, num_neurons * sizeof(NeuronParameters));
    // Set up spikes, keep pointer to recent spikes.
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * num_neurons];

    cudaCheckError("Failed to allocate memory on device for neuron state!");

    // Make temporary arrays for initialization
    float *temp_current = (float*)malloc(num_neurons * sizeof(float));
    float *temp_voltage = (float*)malloc(num_neurons * sizeof(float));
    float *temp_recovery = (float*)malloc(num_neurons * sizeof(float));
    int *temp_spike = (int*)calloc(num_neurons, HISTORY_SIZE * sizeof(int));
    NeuronParameters *temp_params =
        (NeuronParameters*)malloc(num_neurons * sizeof(NeuronParameters));

    if (temp_current == NULL or temp_voltage == NULL or temp_recovery == NULL
            or temp_spike == NULL or temp_params == NULL)
        throw "Failed to allocate space on host for temporary local copies of neuron state!";

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        NeuronParameters &params = model.neuron_parameters[i];
        temp_params[i] = params;
        temp_current[i] = 0;
        temp_voltage[i] = params.c;
        temp_recovery[i] = params.b * params.c;
    }

    // Copy values to GPU
    cudaMemcpy(this->current, temp_current,
        num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->voltage, temp_voltage,
        num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->recovery, temp_recovery,
        num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->spikes, temp_spike,
        num_neurons * HISTORY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->neuron_parameters, temp_params,
        num_neurons * sizeof(NeuronParameters), cudaMemcpyHostToDevice);

    cudaCheckError("Failed to allocate memory on device for neuron state!");

    // Free up temporary memory
    free(temp_current);
    free(temp_voltage);
    free(temp_recovery);
    free(temp_spike);
    free(temp_params);

#else
    // Then initialize actual arrays
    this->current = (float*)malloc(num_neurons * sizeof(float));
    this->voltage = (float*)malloc(num_neurons * sizeof(float));
    this->recovery = (float*)malloc(num_neurons * sizeof(float));
    this->neuron_parameters =
        (NeuronParameters*)malloc(num_neurons * sizeof(NeuronParameters));

    // Set up spikes array
    // Keep track of pointer to least significant word for convenience
    this->spikes = (int*)calloc(num_neurons, HISTORY_SIZE * sizeof(int));
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * num_neurons];

    if (this->current == NULL or this->voltage == NULL
            or this->recovery == NULL or this->spikes == NULL)
        throw "Failed to allocate space on host for neuron state!";

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        NeuronParameters &params = model.neuron_parameters[i];
        this->neuron_parameters[i] = params;
        this->current[i] = 0;
        this->voltage[i] = params.c;
        this->recovery[i] = params.b * params.c;
    }
#endif
}

void State::set_current(int offset, int size, float* input) {
#ifdef PARALLEL
    // Send to GPU
    void* current = &this->current[offset];
    cudaMemcpy(current, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("Failed to set current!");
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = input[nid];
    }
#endif
}

void State::randomize_current(int offset, int size, float max) {
#ifdef PARALLEL
    // Create temporary random array
    float* temp = (float*)malloc(size * sizeof(float));
    if (temp == NULL)
        throw "Failed to allocate memory on host for temporary currents!";

    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = fRand(0, max);
    }

    // Send to GPU
    this->set_current(offset, size, temp);
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = fRand(0, max);
    }
#endif
}

void State::clear_current(int offset, int size) {
#ifdef PARALLEL
    // Create temporary blank array
    float* temp = (float*)malloc(size * sizeof(float));
    if (temp == NULL)
        throw "Failed to allocate memory on host for temporary currents!";

    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = 0.0;
    }

    // Send to GPU
    this->set_current(offset, size, temp);
    free(temp);
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = 0.0;
    }
#endif
}


int* State::get_spikes() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->local_spikes, this->recent_spikes,
        this->num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy spikes from device to host!");
    return this->local_spikes;
#else
    return this->recent_spikes;
#endif
}

float* State::get_current() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->local_current, this->current,
        this->num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy currents from device to host!");
    return this->local_current;
#else
    return this->current;
#endif
}
