#include <math.h>

#include "engine/kernel/attribute_kernel.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"
#include "state/hodgkin_huxley_attributes.h"
#include "util/error_manager.h"

void get_attribute_kernel(ATTRIBUTE_KERNEL *dest, std::string engine_name) {
    if (engine_name == "izhikevich")
        *dest = iz_update_attributes;
    else if (engine_name == "rate_encoding")
        *dest = re_update_attributes;
    else if (engine_name == "hodgkin_huxley")
        *dest = hh_update_attributes;
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized engine type!");
}

/******************************************************************************/
/****************************** IZHIKEVICH ************************************/
/******************************************************************************/

/* Voltage threshold for neuron spiking. */
#define IZ_SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define IZ_EULER_RES 2

/* Milliseconds per timestep */
#define IZ_TIMESTEP_MS 1

GLOBAL void iz_update_attributes(Attributes *att, int start_index, int count) {
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

/******************************************************************************/
/**************************** RATE ENCODING ***********************************/
/******************************************************************************/

GLOBAL void re_update_attributes(Attributes *att, int start_index, int count) {
    float *outputs = (float*)att->output;
    float *inputs = (float*)att->input;
    int total_neurons = att->total_neurons;

#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        nid += start_index;
#else
    for (int nid = start_index ; nid < start_index+count; ++nid) {
#endif
        float next_value = outputs[nid];
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            float curr_value = next_value;
            next_value = outputs[total_neurons * (index + 1) + nid];
            outputs[total_neurons * index + nid] = next_value;
        }
        float input = inputs[nid];
        outputs[total_neurons * index + nid] =
            (input > 0.0) ? tanh(0.1*input) : 0.0;
    }
}

/******************************************************************************/
/**************************** HODGKIN-HUXLEY **********************************/
/******************************************************************************/

/* Voltage threshold for neuron spiking. */
#define HH_SPIKE_THRESH 30.0

/* Euler resolution for voltage update. */
#define HH_RESOLUTION 30
#define HH_TIMESTEPS 30

#define HH_GNABAR 120.0
#define HH_VNA 50.0
#define HH_GKBAR 36.0
#define HH_VK -77.0
#define HH_GL 0.3
#define HH_VL -54.4
#define HH_CM 1.0

GLOBAL void hh_update_attributes(Attributes *att, int start_index, int count) {
    HodgkinHuxleyAttributes *hh_att = (HodgkinHuxleyAttributes*)att;
    float *voltages = hh_att->voltage;
    float *hs = hh_att->h;
    float *ms = hh_att->m;
    float *ns = hh_att->n;
    float *currents = hh_att->current;
    float *current_traces = hh_att->current_trace;
    unsigned int *spikes = hh_att->spikes;
    int total_neurons = hh_att->total_neurons;
    HodgkinHuxleyParameters *params = hh_att->neuron_parameters;

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
        float current =
             (currents[nid] / HH_RESOLUTION) +
            (current_traces[nid] * 0.1);

        float voltage = voltages[nid];
        float h = hs[nid];
        float m = ms[nid];
        float n = ns[nid];
        float iapp = params[nid].iapp;

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        bool already_spiked = voltage > HH_SPIKE_THRESH;
        unsigned int spike = 0;
        for (int i = 0 ; i < HH_TIMESTEPS ; ++i) {
            if (not spike and not already_spiked) m += current / HH_RESOLUTION;
            float am   = 0.1*(voltage+40.0)/( 1.0 - expf(-(voltage+40.0)/10.0) );
            float bm   = 4.0*expf(-(voltage+65.0)/18.0);
            float minf = am/(am+bm);
            float taum = 1.0/(am+bm);

            float ah   = 0.07*expf(-(voltage+65.0)/20.0);
            float bh   = 1.0/( 1.0 + expf(-(voltage+35.0)/10.0) );
            float hinf = ah/(ah+bh);
            float tauh = 1.0/(ah+bh);

            float an   = 0.01*(voltage + 55.0)/(1.0 - expf(-(voltage + 55.0)/10.0));
            float bn   = 0.125*expf(-(voltage + 65.0)/80.0);
            float ninf = an/(an+bn);
            float taun = 1.0/(an+bn);

            float ina = HH_GNABAR * (m*m*m) * h * (voltage-HH_VNA);
            float ik  = HH_GKBAR * (n*n*n*n) * (voltage-HH_VK);
            float il  = HH_GL * (voltage-HH_VL);

            voltage += (iapp - ina - ik - il ) / (HH_RESOLUTION * HH_CM);
            if (voltage != voltage) voltage = HH_SPIKE_THRESH+1;
            else {
                h +=  (hinf - h) / (HH_RESOLUTION * tauh);
                n +=  (ninf - n) / (HH_RESOLUTION * taun);
                m +=  (minf - m) / (HH_RESOLUTION * taum);
            }

            spike = spike or voltage > HH_SPIKE_THRESH;
        }
        spike = spike and not already_spiked;

        hs[nid] = h;
        ms[nid] = m;
        ns[nid] = n;
        voltages[nid] = voltage;
        current_traces[nid] = current;

        /********************
         *** SPIKE UPDATE ***
         ********************/
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
    }
}
