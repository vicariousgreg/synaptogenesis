#include <cstdlib>
#include <cstdio>
#include <string>

#include "state/izhikevich_attributes.h"
#include "engine/izhikevich_kernel.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

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

static IzhikevichParameters create_parameters(std::string str) {
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
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognizer parameter string: " + str);
}

IzhikevichAttributes::IzhikevichAttributes(Model* model) : Attributes(model, BIT) {
    float* local_voltage = (float*) allocate_host(total_neurons, sizeof(float));
    float* local_recovery = (float*) allocate_host(total_neurons, sizeof(float));
    IzhikevichParameters* local_params =
        (IzhikevichParameters*) allocate_host(total_neurons, sizeof(IzhikevichParameters));

    // Fill in table
    for (int i = 0; i < model->all_layers.size(); ++i) {
        Layer *layer = model->all_layers[i];
        IzhikevichParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j) {
            local_params[layer->index+j] = params;
            local_voltage[layer->index+j] = params.c;
            local_recovery[layer->index+j] = params.b * params.c;
        }
    }

#ifdef PARALLEL
    // Allocate space on GPU and copy data
    this->voltage = (float*)
        allocate_device(total_neurons, sizeof(float), local_voltage);
    this->recovery = (float*)
        allocate_device(total_neurons, sizeof(float), local_recovery);
    this->neuron_parameters = (IzhikevichParameters*)
        allocate_device(total_neurons, sizeof(IzhikevichParameters), local_params);
    free(local_voltage);
    free(local_recovery);
    free(local_params);
#else
    this->voltage = local_voltage;
    this->recovery = local_recovery;
    this->neuron_parameters = local_params;
#endif
}

IzhikevichAttributes::~IzhikevichAttributes() {
#ifdef PARALLEL
    cudaFree(this->voltage);
    cudaFree(this->recovery);
    cudaFree(this->neuron_parameters);
#else
    free(this->voltage);
    free(this->recovery);
    free(this->neuron_parameters);
#endif
}

#ifdef PARALLEL
void IzhikevichAttributes::update(int start_index, int count, cudaStream_t &stream) {
    int threads = calc_threads(count);
    int blocks = calc_blocks(count);
    izhikevich<<<blocks, threads, 0, stream>>>(
#else
void IzhikevichAttributes::update(int start_index, int count) {
    izhikevich(
#endif
        voltage,
        recovery,
        input,
        neuron_parameters,
        start_index, count);
#ifdef PARALLEL
    cudaCheckError("Failed to update neuron voltages!");

    calc_spikes<<<blocks, threads, 0, stream>>>(
#else
    calc_spikes(
#endif
        (int*) output,
        voltage,
        recovery,
        neuron_parameters,
        start_index, count, total_neurons);
#ifdef PARALLEL
    cudaCheckError("Failed to timestep spikes!");
#endif
}
