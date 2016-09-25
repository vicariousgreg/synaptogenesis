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

/******************************************************************************
 **************************** INITIALIZATION **********************************
 ******************************************************************************/

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

    // Local spikes for output reporting
    this->spikes = (int*) allocate_host(num_neurons, sizeof(int));
    this->current = (float*) allocate_host(num_neurons, sizeof(float));
    this->voltage = (float*) allocate_host(num_neurons, sizeof(float));
    this->recovery = (float*) allocate_host(num_neurons, sizeof(float));
    this->neuron_parameters =
        (IzhikevichParameters*) allocate_host(num_neurons, sizeof(IzhikevichParameters));

    // Keep pointer to recent spikes.
    this->recent_spikes = &this->spikes[(HISTORY_SIZE-1) * num_neurons];

    // Fill in table
    for (int i = 0 ; i < num_neurons ; ++i) {
        std::string &param_string = model->neuron_parameters[i];
        IzhikevichParameters params = create_parameters(param_string);
        this->neuron_parameters[i] = params;
        this->current[i] = 0;
        this->voltage[i] = params.c;
        this->recovery[i] = params.b * params.c;
    }

#ifdef PARALLEL
    // Allocate space on GPU and copy data
    this->device_current = (float*)
        allocate_device(num_neurons, sizeof(float), this->current);
    this->device_voltage = (float*)
        allocate_device(num_neurons, sizeof(float), this->voltage);
    this->device_recovery = (float*)
        allocate_device(num_neurons, sizeof(float), this->recovery);
    this->device_spikes = (int*)
        allocate_device(num_neurons * HISTORY_SIZE, sizeof(int), this->spikes);
    this->device_neuron_parameters = (IzhikevichParameters*)
        allocate_device(num_neurons, sizeof(IzhikevichParameters), this->neuron_parameters);
    this->device_recent_spikes = &this->device_spikes[(HISTORY_SIZE-1) * num_neurons];
#endif
    this->weight_matrices = build_weight_matrices(model, 1);
}

/******************************************************************************
 ************************ TIMESTEP DYNAMICS ***********************************
 ******************************************************************************/

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


/******************************************************************************
 *************************** STATE INTERACTION ********************************
 ******************************************************************************/

void Izhikevich::set_current(int layer_id, float* input) {
    int offset = this->model->layers[layer_id].index;
    int size = this->model->layers[layer_id].size;
#ifdef PARALLEL
    // Send to GPU
    cudaMemcpy(&this->device_current[offset], input,
        size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("Failed to set current!");
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = input[nid];
    }
#endif
}

void Izhikevich::randomize_current(int layer_id, float max) {
    int size = this->model->layers[layer_id].size;
#ifdef PARALLEL
    // Create temporary random array
    float* temp = (float*)malloc(size * sizeof(float));
    if (temp == NULL)
        throw "Failed to allocate memory on host for temporary currents!";

    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = fRand(0, max);
    }

    // Send to GPU
    this->set_current(layer_id, temp);
#else
    int offset = this->model->layers[layer_id].index;
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = fRand(0, max);
    }
#endif
}

void Izhikevich::clear_current(int layer_id) {
    int size = this->model->layers[layer_id].size;
#ifdef PARALLEL
    // Create temporary blank array
    float* temp = (float*)malloc(size * sizeof(float));
    if (temp == NULL)
        throw "Failed to allocate memory on host for temporary currents!";

    for (int nid = 0 ; nid < size; ++nid) {
        temp[nid] = 0.0;
    }

    // Send to GPU
    this->set_current(layer_id, temp);
    free(temp);
#else
    int offset = this->model->layers[layer_id].index;
    for (int nid = 0 ; nid < size; ++nid) {
        this->current[nid+offset] = 0.0;
    }
#endif
}


int* Izhikevich::get_spikes() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->spikes, this->device_spikes,
        this->model->num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy spikes from device to host!");
#endif
    return this->spikes;
}

float* Izhikevich::get_current() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->current, this->device_current,
        this->model->num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy currents from device to host!");
#endif
    return this->current;
}
