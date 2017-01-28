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
    int *spikes = iz_att->spikes;
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
        int spike = voltage >= IZ_SPIKE_THRESH;

        // Reduce reads, chain values.
        int curr_value, new_value;
        int next_value = spikes[nid];

        // Shift all the bits.
        // Check if next word is negative (1 for MSB).
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            curr_value = next_value;
            next_value = spikes[total_neurons * (index + 1) + nid];

            // Shift bits, carry over MSB from next value.
            new_value = (curr_value << 1) + (next_value < 0);
            spikes[total_neurons*index + nid] = new_value;
        }

        // Least significant value already loaded into next_value.
        // Index moved appropriately from loop.
        spikes[total_neurons*index + nid] = (next_value << 1) + spike;

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
        float curr_value, next_value = outputs[nid];
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            curr_value = next_value;
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
#define HH_RESOLUTION 100
#define HH_TIMESTEPS 10

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
    int *spikes = hh_att->spikes;
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
        float current = currents[nid] / HH_RESOLUTION;
        double current_trace = current_traces[nid] * 0.5 + current;

        double voltage = voltages[nid];
        double h = hs[nid];
        double m = ms[nid];
        double n = ns[nid];
        double iapp = params[nid].iapp;

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        int spike = 0;
        for (int i = 0 ; i < HH_TIMESTEPS ; ++i) {
            m += current / HH_TIMESTEPS;
            double am   = 0.1*(voltage+40.0)/( 1.0 - exp(-(voltage+40.0)/10.0) );
            double bm   = 4.0*exp(-(voltage+65.0)/18.0);
            double minf = am/(am+bm);
            double taum = 1.0/(am+bm);

            double ah   = 0.07*exp(-(voltage+65.0)/20.0);
            double bh   = 1.0/( 1.0 + exp(-(voltage+35.0)/10.0) );
            double hinf = ah/(ah+bh);
            double tauh = 1.0/(ah+bh);

            double an   = 0.01*(voltage + 55.0)/(1.0 - exp(-(voltage + 55.0)/10.0));
            double bn   = 0.125*exp(-(voltage + 65.0)/80.0);
            double ninf = an/(an+bn);
            double taun = 1.0/(an+bn);

            double ina = HH_GNABAR * (m*m*m) * h * (voltage-HH_VNA);
            double ik  = HH_GKBAR * (n*n*n*n) * (voltage-HH_VK);
            double il  = HH_GL * (voltage-HH_VL);

            voltage += (iapp - ina - ik - il ) / (HH_RESOLUTION * HH_CM);
            h +=  (hinf - h) / (HH_RESOLUTION * tauh);
            n +=  (ninf - n) / (HH_RESOLUTION * taun);
            m +=  (minf - m) / (HH_RESOLUTION * taum);

            if (voltage >= HH_SPIKE_THRESH) {
                spike = 1;
                voltage = HH_SPIKE_THRESH;
            }
        }

        hs[nid] = h;
        ms[nid] = m;
        ns[nid] = n;
        voltages[nid] = voltage;
        current_traces[nid] = current;

        /********************
         *** SPIKE UPDATE ***
         ********************/
        // Reduce reads, chain values.
        int curr_value, new_value;
        int next_value = spikes[nid];

        // Shift all the bits.
        // Check if next word is negative (1 for MSB).
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            curr_value = next_value;
            next_value = spikes[total_neurons * (index + 1) + nid];

            // Shift bits, carry over MSB from next value.
            new_value = (curr_value << 1) + (next_value < 0);
            spikes[total_neurons*index + nid] = new_value;
        }

        // Least significant value already loaded into next_value.
        // Index moved appropriately from loop.
        spikes[total_neurons*index + nid] = (next_value << 1) + spike;
    }
}
