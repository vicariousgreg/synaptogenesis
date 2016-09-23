#ifdef parallel

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#endif

#include <cstdlib>
#include "state.h"
#include "tools.h"

void State::build(int num_neurons, NeuronParameters* neuron_parameters) {
    this->num_neurons = num_neurons;

#ifdef parallel
    // Local spikes for output reporting
    this->local_spikes = (int*)calloc(num_neurons, sizeof(int));
    this->local_current = (float*)calloc(num_neurons, sizeof(float));

    // Allocate space on GPU
    cudaMalloc(&this->current, num_neurons * sizeof(float));
    cudaMalloc(&this->voltage, num_neurons * sizeof(float));
    cudaMalloc(&this->recovery, num_neurons * sizeof(float));

    // Set up spikes, keep pointer to recent spikes.
    cudaMalloc(&this->spikes, HISTORY_SIZE * num_neurons * sizeof(int));
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * num_neurons];

    // Make temporary arrays for initialization
    float *temp_current = (float*)malloc(num_neurons * sizeof(float));
    float *temp_voltage = (float*)malloc(num_neurons * sizeof(float));
    float *temp_recovery = (float*)malloc(num_neurons * sizeof(float));
    int *temp_spike = (int*)calloc(num_neurons, HISTORY_SIZE * sizeof(int));

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        NeuronParameters &params = neuron_parameters[i];
        temp_current[i] = 0;
        temp_voltage[i] = params.c;
        temp_recovery[i] = params.b * params.c;
    }

    // Copy values to GPU
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(this->current, temp_current,
        num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->voltage, temp_voltage,
        num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->recovery, temp_recovery,
        num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->spikes, temp_spike,
        num_neurons * HISTORY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaCheckError();

    // Free up temporary memory
    free(temp_current);
    free(temp_voltage);
    free(temp_recovery);
    free(temp_spike);

#else
    // Then initialize actual arrays
    this->current = (float*)malloc(num_neurons * sizeof(float));
    this->voltage = (float*)malloc(num_neurons * sizeof(float));
    this->recovery = (float*)malloc(num_neurons * sizeof(float));

    // Set up spikes array
    // Keep track of pointer to least significant word for convenience
    this->spikes = (int*)calloc(num_neurons, HISTORY_SIZE * sizeof(int));
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * num_neurons];

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        NeuronParameters &params = neuron_parameters[i];
        this->current[i] = 0;
        this->voltage[i] = params.c;
        this->recovery[i] = params.b * params.c;
    }
#endif
}

void State::set_current(int offset, int size, float* input) {
#ifdef parallel
    // Send to GPU
    void* current = &this->current[offset];
    cudaMemcpy(current, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaCheckError();
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = input[nid];
    }
#endif
}

void State::randomize_current(int offset, int size, float max) {
#ifdef parallel
    // Send to GPU
    float* temp = (float*)malloc(size * sizeof(float));
    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = fRand(0, max);
    }
    this->set_current(offset, size, temp);
    free(temp);
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = fRand(0, max);
    }
#endif
}

void State::clear_current(int offset, int size) {
#ifdef parallel
    // Send to GPU
    float* temp = (float*)malloc(size * sizeof(float));
    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = 0.0;
    }
    this->set_current(offset, size, temp);
    free(temp);
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = 0.0;
    }
#endif
}


// GETTERS
int* State::get_spikes() {
#ifdef parallel
    // Copy from GPU to local location
    cudaMemcpy(this->local_spikes, this->recent_spikes,
        this->num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();
    return this->local_spikes;
#else
    return this->recent_spikes;
#endif
}

float* State::get_current() {
#ifdef parallel
    // Copy from GPU to local location
    cudaMemcpy(this->local_current, this->current,
        this->num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();
    return this->local_current;
#else
    return this->current;
#endif
}
