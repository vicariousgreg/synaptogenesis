#include <math.h>

#include "engine/kernel/attribute_kernel.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"

void get_attribute_kernel(ATTRIBUTE_KERNEL *dest, std::string engine_name) {
    if (engine_name == "izhikevich")
        *dest = iz_update_attributes;
    else if (engine_name == "rate_encoding")
        *dest = re_update_attributes;
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized engine type!");
}

/******************************************************************************/
/****************************** IZHIKEVICH ************************************/
/******************************************************************************/

/* Voltage threshold for neuron spiking. */
#define SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define EULER_RES 2

GLOBAL void iz_update_attributes(Attributes *att,
        int start_index, int count, int total_neurons) {
    IzhikevichAttributes *iz_att = (IzhikevichAttributes*)att;
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
        float voltage = iz_att->voltage[nid];
        float recovery = iz_att->recovery[nid];
        float current = iz_att->current[nid];

        float a = iz_att->neuron_parameters[nid].a;
        float b = iz_att->neuron_parameters[nid].b;

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        for (int i = 0 ; i < EULER_RES && voltage < SPIKE_THRESH ; ++i) {
            float delta_v = (0.04 * voltage * voltage) +
                            (5*voltage) + 140 - recovery + current;
            voltage += delta_v / EULER_RES;
            recovery += a * ((b * voltage) - recovery) / EULER_RES;
        }

        /********************
         *** SPIKE UPDATE ***
         ********************/
        // Determine if spike occurred
        int spike = voltage >= SPIKE_THRESH;

        // Reduce reads, chain values.
        int curr_value, new_value;
        int next_value = iz_att->spikes[nid];

        // Shift all the bits.
        // Check if next word is negative (1 for MSB).
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            curr_value = next_value;
            next_value = iz_att->spikes[total_neurons * (index + 1) + nid];

            // Shift bits, carry over MSB from next value.
            new_value = (curr_value << 1) + (next_value < 0);
            iz_att->spikes[total_neurons*index + nid] = new_value;
        }

        // Least significant value already loaded into next_value.
        // Index moved appropriately from loop.
        iz_att->spikes[total_neurons*index + nid] = (next_value << 1) + spike;

        // Reset voltage if spiked.
        if (spike) {
            iz_att->voltage[nid] = iz_att->neuron_parameters[nid].c;
            iz_att->recovery[nid] = recovery + iz_att->neuron_parameters[nid].d;
        } else {
            iz_att->voltage[nid] = voltage;
            iz_att->recovery[nid] = recovery;
        }
    }
}

/******************************************************************************/
/**************************** RATE ENCODING ***********************************/
/******************************************************************************/

GLOBAL void re_update_attributes(Attributes *att,
        int start_index, int count, int total_neurons) {
    RateEncodingAttributes *re_att = (RateEncodingAttributes*)att;
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
#else
    for (int nid = start_index ; nid < start_index+count; ++nid) {
#endif
        float curr_value, next_value = re_att->output[nid];
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            curr_value = next_value;
            next_value = re_att->output[total_neurons * (index + 1) + nid];
            re_att->output[total_neurons*index + nid] = next_value;
        }
        float input = re_att->input[nid];
        re_att->output[total_neurons*index + nid] =
            (input > 0.0) ? tanh(0.01*input) : 0.0;
    }
}
