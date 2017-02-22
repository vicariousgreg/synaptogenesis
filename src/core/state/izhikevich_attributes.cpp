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

IzhikevichAttributes::IzhikevichAttributes(Structure* structure)
        : SpikingAttributes(structure) {
    this->recovery = (float*) allocate_host(total_neurons, sizeof(float));
    this->neuron_parameters = (IzhikevichParameters*)
        allocate_host(total_neurons, sizeof(IzhikevichParameters));

    // Fill in table
    int start_index = 0;
    for (auto& layer : structure->get_layers()) {
        IzhikevichParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j) {
            neuron_parameters[start_index+j] = params;
            voltage[start_index+j] = params.c;
            recovery[start_index+j] = params.b * params.c;
        }
        start_index += layer->size;
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

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

/* Voltage threshold for neuron spiking. */
#define IZ_SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define IZ_EULER_RES 2

/* Milliseconds per timestep */
#define IZ_TIMESTEP_MS 1

GLOBAL void iz_attribute_kernel(const Attributes *att, int start_index, int count) {
    IzhikevichAttributes *iz_att = (IzhikevichAttributes*)att;
    float *voltages = iz_att->voltage;
    float *recoveries = iz_att->recovery;
    float *currents = iz_att->current;
    unsigned int *spikes = iz_att->spikes;
    int total_neurons = iz_att->total_neurons;
    IzhikevichParameters *params = iz_att->neuron_parameters;

#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        nid += start_index;
#else
    for (int nid = start_index; nid < start_index+count; ++nid) {
#endif
        /**********************
         *** VOLTAGE UPDATE ***
         **********************/
        float voltage = voltages[nid];
        float recovery = recoveries[nid];
        float current = currents[nid];

        float a = params[nid].a;
        float b = params[nid].b;

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        for (int i = 0 ; i < IZ_TIMESTEP_MS * IZ_EULER_RES && voltage < IZ_SPIKE_THRESH ; ++i) {
            float delta_v = (0.04 * voltage * voltage) +
                            (5*voltage) + 140 - recovery + current;
            voltage += delta_v / IZ_EULER_RES;
            recovery += a * ((b * voltage) - recovery) / IZ_EULER_RES;
        }

        /********************
         *** SPIKE UPDATE ***
         ********************/
        // Determine if spike occurred
        unsigned int spike = voltage >= IZ_SPIKE_THRESH;

        // Reduce reads, chain values.
        unsigned int next_value = spikes[nid];

        // Shift all the bits.
        // Check if next word is odd (1 for LSB).
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            unsigned int curr_value = next_value;
            next_value = spikes[total_neurons * (index + 1) + nid];

            // Shift bits, carry over LSB from next value.
            spikes[total_neurons*index + nid] = (curr_value >> 1) | (next_value << 31);
        }

        // Least significant value already loaded into next_value.
        // Index moved appropriately from loop.
        spikes[total_neurons*index + nid] = (next_value >> 1) | (spike << 31);

        // Reset voltage if spiked.
        if (spike) {
            voltages[nid] = params[nid].c;
            recoveries[nid] = recovery + params[nid].d;
        } else {
            voltages[nid] = voltage;
            recoveries[nid] = recovery;
        }
    }
}
