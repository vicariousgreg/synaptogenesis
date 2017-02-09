#include <string>

#include "state/izhikevich_attributes.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

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
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognizer parameter string: " + str);
}

IzhikevichAttributes::IzhikevichAttributes(Model* model)
        : SpikingAttributes(model) {
    this->recovery = (float*) allocate_host(total_neurons, sizeof(float));
    this->neuron_parameters = (IzhikevichParameters*)
        allocate_host(total_neurons, sizeof(IzhikevichParameters));

    // Fill in table
    for (auto& layer : model->get_layers()) {
        IzhikevichParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j) {
            int start_index = layer->get_start_index();
            neuron_parameters[start_index+j] = params;
            voltage[start_index+j] = params.c;
            recovery[start_index+j] = params.b * params.c;
        }
    }
}

IzhikevichAttributes::~IzhikevichAttributes() {
#ifdef PARALLEL
    cudaFree(this->recovery);
    cudaFree(this->neuron_parameters);
#else
    free(this->recovery);
    free(this->neuron_parameters);
#endif
}

#ifdef PARALLEL
void IzhikevichAttributes::send_to_device() {
    SpikingAttributes::send_to_device();

    // Allocate space on GPU and copy data
    float* device_recovery= (float*)
        allocate_device(total_neurons, sizeof(float), this->recovery);
    IzhikevichParameters* device_params = (IzhikevichParameters*)
        allocate_device(total_neurons, sizeof(IzhikevichParameters),
        this->neuron_parameters);

    free(this->recovery);
    free(this->neuron_parameters);

    this->recovery = device_recovery;
    this->neuron_parameters = device_params;
}
#endif
