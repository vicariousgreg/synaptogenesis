#include <cstdlib>
#include "environment.h"

void dot(int sign, int* spikes, double* weights, double* currents,
         int from_size, int to_index) {
    for (int index = 0 ; index < from_size ; ++index) {
        currents[to_index] += sign * spikes[index] * weights[index];
    }
}

void mult(int sign, int* spikes, double* weights, double* currents,
          int from_size, int to_size) {
    for (int row = 0 ; row < to_size ; ++row) {
        dot(sign, spikes, weights, currents, from_size, row);
    }
}

void izhikevich(double* voltages, double*recoveries, double* currents,
                NeuronParameters* neuron_params, int num_neurons) {
    double voltage, recovery, current, delta_v;
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

void calc_spikes(int* spikes, int* ages, double* voltages, double* recoveries,
                 NeuronParameters* neuron_params, int num_neurons) {
    int spike; 
    NeuronParameters *params;

    /* 4. Timestep */
    // Determine spikes.
    for (int i = 0; i < num_neurons; ++i) {
        spike = voltages[i] >= SPIKE_THRESH;
        spikes[i] = spike;

        // Increment or reset spike ages.
        // Also, reset voltage if spiked.
        if (spike) {
            params = &neuron_params[i];
            ages[i] = 0;
            voltages[i] = params->c;
            recoveries[i] += params->d;
        } else {
            ++ages[i];
        }
    }
}
