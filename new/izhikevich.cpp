#include <cstdlib>
#include <cstdio>
#include <string>

#include "izhikevich.h"
#include "model.h"
#include "tools.h"
#include "constants.h"
#include "izhikevich_operations.h"
#include "parallel.h"

#define DEF_PARAM(name, a,b,c,d) \
    static const IzhikevichParameters name = IzhikevichParameters(a,b,c,d);

/* Izhikevich Parameters Table */
DEF_PARAM(DEFAULT          , 0.02, 0.2 , -70.0, 2   ); // Default
DEF_PARAM(REGULAR          , 0.02, 0.2 , -65.0, 8   ); // Regular Spiking
DEF_PARAM(BURSTING         , 0.02, 0.2 , -55.0, 4   ); // Intrinsically Bursting
DEF_PARAM(CHATTERING       , 0.02, 0.2 , -50.0, 2   ); // Chattering
DEF_PARAM(FAST             , 0.1 , 0.2 , -65.0, 2   ); // Fast Spiking
DEF_PARAM(LOW_THRESHOLD    , 0.02, 0.25, -65.0, 2   ); // Low Threshold
DEF_PARAM(THALAMO_CORTICAL , 0.02, 0.25, -65.0, 0.05); // Thalamo-cortical
DEF_PARAM(RESONATOR        , 0.1 , 0.26, -65.0, 2   ); // Resonator
DEF_PARAM(PHOTORECEPTOR    , 0   , 0   , -82.6, 0   ); // Photoreceptor
DEF_PARAM(HORIZONTAL       , 0   , 0   , -82.6, 0   ); // Horizontal Cell

IzhikevichParameters create_parameters(std::string str) {
    if (str == "random positive") {
        // (ai; bi) = (0:02; 0:2) and (ci; di) = (-65; 8) + (15;-6)r2
        float a = 0.02;
        float b = 0.2; // increase for higher frequency oscillations

        float rand = fRand(0, 1);
        float c = -65.0 + (15.0 * rand * rand);

        rand = fRand(0, 1);
        float d = 8.0 - (6.0 * (rand * rand));
        return IzhikevichParameters(a,b,c,d);
    } else if (str == "random negative") {
        //(ai; bi) = (0:02; 0:25) + (0:08;-0:05)ri and (ci; di)=(-65; 2).
        float rand = fRand(0, 1);
        float a = 0.02 + (0.08 * rand);

        rand = fRand(0, 1);
        float b = 0.25 - (0.05 * rand);

        float c = -65.0;
        float d = 2.0;
        return IzhikevichParameters(a,b,c,d);
    }
    else if (str == "default")            return DEFAULT;
    else if (str == "regular")            return REGULAR;
    else if (str == "bursting")           return BURSTING;
    else if (str == "chattering")         return CHATTERING;
    else if (str == "fast")               return FAST;
    else if (str == "low_threshold")      return LOW_THRESHOLD;
    else if (str == "thalamo_cortical")   return THALAMO_CORTICAL;
    else if (str == "resonator")          return RESONATOR;
    else if (str == "photoreceptor")      return PHOTORECEPTOR;
    else if (str == "horizontal")         return HORIZONTAL;
    else throw ("Unrecognized parameter string: " + str).c_str();
}

void Izhikevich::build(Model* model) {
    this->model = model;
    int num_neurons = model->num_neurons;

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
    cudaMalloc(&this->neuron_parameters, num_neurons * sizeof(IzhikevichParameters));
    // Set up spikes, keep pointer to recent spikes.
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * num_neurons];

    cudaCheckError("Failed to allocate memory on device for neuron state!");

    // Make temporary arrays for initialization
    float *temp_current = (float*)malloc(num_neurons * sizeof(float));
    float *temp_voltage = (float*)malloc(num_neurons * sizeof(float));
    float *temp_recovery = (float*)malloc(num_neurons * sizeof(float));
    int *temp_spike = (int*)calloc(num_neurons, HISTORY_SIZE * sizeof(int));
    IzhikevichParameters *temp_params =
        (IzhikevichParameters*)malloc(num_neurons * sizeof(IzhikevichParameters));

    if (temp_current == NULL or temp_voltage == NULL or temp_recovery == NULL
            or temp_spike == NULL or temp_params == NULL)
        throw "Failed to allocate space on host for temporary local copies of neuron state!";

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        std::string &param_string = model->neuron_parameters[i];
        IzhikevichParameters params = create_parameters(param_string);
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
        num_neurons * sizeof(IzhikevichParameters), cudaMemcpyHostToDevice);

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
        (IzhikevichParameters*)malloc(num_neurons * sizeof(IzhikevichParameters));

    // Set up spikes array
    // Keep track of pointer to least significant word for convenience
    this->spikes = (int*)calloc(num_neurons, HISTORY_SIZE * sizeof(int));
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * num_neurons];

    if (this->current == NULL or this->voltage == NULL
            or this->recovery == NULL or this->spikes == NULL)
        throw "Failed to allocate space on host for neuron state!";

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        std::string &param_string = model->neuron_parameters[i];
        IzhikevichParameters params = create_parameters(param_string);
        this->neuron_parameters[i] = params;
        this->current[i] = 0;
        this->voltage[i] = params.c;
        this->recovery[i] = params.b * params.c;
    }
#endif
    this->weight_matrices = build_weight_matrices(model, 1);
}


// TODO: implement me
void Izhikevich::step_input() {
    float* current = this->current;
    int* spikes = this->spikes;

    /* 2. Activation */
    // For each weight matrix...
    //   Update Currents using synaptic input
    //     current = operation ( current , dot ( spikes * weights ) )
    //
    // TODO: optimize order, create batches of parallelizable computations,
    //       and move cuda barriers around batches
    for (int cid = 0 ; cid < this->model->num_connections; ++cid) {
        update_currents(
            this->model->connections[cid], this->weight_matrices[cid],
            spikes, current, this->model->num_neurons);
    }
}

void Izhikevich::step_state() {
    /* 3. Voltage Updates */
    update_voltages(
        this->voltage,
        this->recovery,
        this->current,
        this->neuron_parameters,
        this->model->num_neurons);
}

void Izhikevich::step_output() {
    /* 4. Timestep */
    update_spikes(
        this->spikes,
        this->voltage,
        this->recovery,
        this->neuron_parameters,
        this->model->num_neurons);
}

void Izhikevich::step_weights() {
    /* 5. Update weights */
    update_weights();
}




void Izhikevich::set_current(int offset, int size, float* input) {
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

void Izhikevich::randomize_current(int offset, int size, float max) {
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

void Izhikevich::clear_current(int offset, int size) {
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


int* Izhikevich::get_spikes() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->local_spikes, this->recent_spikes,
        this->model->num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy spikes from device to host!");
    return this->local_spikes;
#else
    return this->recent_spikes;
#endif
}

float* Izhikevich::get_current() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->local_current, this->current,
        this->model->num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy currents from device to host!");
    return this->local_current;
#else
    return this->current;
#endif
}
