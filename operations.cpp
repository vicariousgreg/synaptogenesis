#include <cstdlib>
#include <climits>
#include "environment.h"

#ifdef parallel

__global__ void mult(int sign, int* spikes, float* weights, float* currents,
            int from_size, int to_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < to_size) {
        float sum = 0;
        for (int row = 0 ; row < from_size ; ++row) {
            sum += (spikes[row] % 2) * weights[row * to_size + col];
        }
        currents[col] += sign * sum;
    }
}

/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void izhikevich(float* voltages, float*recoveries,
        float* currents, NeuronParameters* neuron_params, int num_neurons) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;

    /* 3. Voltage Updates */
    // For each neuron...
    //   update voltage according to current
    //     Euler's method with #defined resolution
    if (nid < num_neurons) {
        float voltage = voltages[nid];
        float recovery = recoveries[nid];
        float current = currents[nid];
        float delta_v = 0;

        NeuronParameters *params = &neuron_params[nid];

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        for (int i = 0 ; i < EULER_RES && voltage < SPIKE_THRESH ; ++i) {
            delta_v = (0.04 * voltage * voltage) +
                            (5*voltage) + 140 - recovery + current;
            voltage += delta_v / EULER_RES;
            //recovery += params->a * ((params->b * voltage) - recovery)
            //                / EULER_RES;
        }

        recovery += params->a * ((params->b * voltage) - recovery);
        voltages[nid] = voltage;
        recoveries[nid] = recovery;
    }
}

/* Parallel implementation of spike update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
__global__ void calc_spikes(int* spikes, float* voltages,
        float* recoveries, NeuronParameters* neuron_params, int num_neurons) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;

    /* 4. Timestep */
    // Determine spikes.
    if (nid < num_neurons) {
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
            NeuronParameters *params = &neuron_params[nid];
            voltages[nid] = params->c;
            recoveries[nid] += params->d;
        }
    }
}

#else

void mult(int sign, int* spikes, float* weights, float* currents,
          int from_size, int to_size) {
    for (int row = 0 ; row < from_size ; ++row) {
        for (int col = 0 ; col < to_size ; ++col) {
            currents[col] += sign * (spikes[row] % 2) * weights[row*to_size + col];
        }
    }
}

void izhikevich(float* voltages, float*recoveries, float* currents,
                NeuronParameters* neuron_params, int num_neurons) {
    float voltage, recovery, current, delta_v;
    NeuronParameters *params;

    /* 3. Voltage Updates */
    // For each neuron...
    //   update voltage according to current
    //     Euler's method with #defined resolution
    for (int nid = 0 ; nid < num_neurons; ++nid) {
        voltage = voltages[nid];
        recovery = recoveries[nid];
        current = currents[nid];
        delta_v = 0;

        params = &neuron_params[nid];

        // Euler's method for voltage/recovery update
        // If the voltage exceeds the spiking threshold, break
        for (int i = 0 ; i < EULER_RES && voltage < SPIKE_THRESH ; ++i) {
            delta_v = (0.04 * voltage * voltage) +
                            (5*voltage) + 140 - recovery + current;
            voltage += delta_v / EULER_RES;
            //recovery += params->a * ((params->b * voltage) - recovery)
            //                / EULER_RES;
        }
        recovery += params->a * ((params->b * voltage) - recovery);
        voltages[nid] = voltage;
        recoveries[nid] = recovery;
    }
}

void calc_spikes(int* spikes, float* voltages, float* recoveries,
                 NeuronParameters* neuron_params, int num_neurons) {
    int spike, new_value; 
    NeuronParameters *params;

    /* 4. Timestep */
    // Determine spikes.
    for (int nid = 0; nid < num_neurons; ++nid) {
        spike = voltages[nid] >= SPIKE_THRESH;

        // Shift all the bits.
        // Check if next word is negative (1 for MSB).
        for (int index = 0 ; index < HISTORY_SIZE-1 ; ++ index) {
            new_value = spikes[num_neurons*index + nid] << 1
                + (spikes[num_neurons*(index+1) + nid] < 0);
            spikes[num_neurons*index + nid] = new_value;
        }
        new_value = (spikes[num_neurons*(HISTORY_SIZE-1) + nid] << 1) + spike;
        spikes[num_neurons*(HISTORY_SIZE-1) + nid] = new_value;

        // Reset voltage if spiked.
        if (spike) {
            params = &neuron_params[nid];
            voltages[nid] = params->c;
            recoveries[nid] += params->d;
        }
    }
}

#endif
