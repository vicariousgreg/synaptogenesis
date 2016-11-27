#include "engine/kernel/izhikevich_kernel.h"

/* Voltage threshold for neuron spiking. */
#define SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define EULER_RES 2

GLOBAL void izhikevich(float* voltages, float* recoveries,
        float* currents, IzhikevichParameters* neuron_params,
        int start_index, int count) {
    float voltage, recovery, current, delta_v;
    IzhikevichParameters *params;

    /* 3. Voltage Updates */
    // For each neuron...
    //   update voltage according to current
    //     Euler's method with #defined resolution
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        nid += start_index;
#else
    for (int nid = start_index ; nid < count+start_index; ++nid) {
#endif
        float voltage = voltages[nid];
        float recovery = recoveries[nid];
        float current = currents[nid];
        float delta_v = 0;

        IzhikevichParameters *params = &neuron_params[nid];

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        for (int i = 0 ; i < EULER_RES && voltage < SPIKE_THRESH ; ++i) {
            delta_v = (0.04 * voltage * voltage) +
                            (5*voltage) + 140 - recovery + current;
            voltage += delta_v / EULER_RES;
            recovery += params->a * ((params->b * voltage) - recovery)
                            / EULER_RES;
        }

        //recovery += params->a * ((params->b * voltage) - recovery);
        voltages[nid] = voltage;
        recoveries[nid] = recovery;
    }
}

GLOBAL void calc_spikes(int* spikes, float* voltages,
        float* recoveries, IzhikevichParameters* neuron_params,
        int start_index, int count, int num_neurons) {
    int spike, new_value; 
    IzhikevichParameters *params;

    /* 4. Timestep */
    // Determine spikes.
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        nid += start_index;
#else
    for (int nid = start_index; nid < start_index+count; ++nid) {
#endif
        int spike = voltages[nid] >= SPIKE_THRESH;

        // Reduce reads, chain values.
        int curr_value, new_value;
        int next_value = spikes[nid];

        // Shift all the bits.
        // Check if next word is negative (1 for MSB).
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++ index) {
            curr_value = next_value;
            next_value = spikes[num_neurons * (index + 1) + nid];

            // Shift bits, carry over MSB from next value.
            new_value = curr_value << 1 + (next_value < 0);
            spikes[num_neurons*index + nid] = new_value;
        }

        // Least significant value already loaded into next_value.
        // Index moved appropriately from loop.
        new_value = (next_value << 1) + spike;
        spikes[num_neurons*index + nid] = new_value;

        // Reset voltage if spiked.
        if (spike) {
            IzhikevichParameters *params = &neuron_params[nid];
            voltages[nid] = params->c;
            recoveries[nid] += params->d;
        }
    }
}
