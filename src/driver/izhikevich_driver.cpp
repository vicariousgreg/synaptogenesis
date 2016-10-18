#include "driver/izhikevich_driver.h"

/* Voltage threshold for neuron spiking. */
#define SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define EULER_RES 10

DEVICE float iz_calc_input(int output, int mask) {
    return output & mask;
}

DEVICE float (*iz_calc_input_ptr)(int, int) = iz_calc_input;

IzhikevichDriver::IzhikevichDriver(Model *model) {
    this->iz_state = new IzhikevichState(model);
    this->state = this->iz_state;
#ifdef PARALLEL
    cudaMemcpyFromSymbol(&this->calc_input_ptr, iz_calc_input_ptr, sizeof(void *));
#else
    this->calc_input_ptr = iz_calc_input_ptr;
#endif

    for (int i = 0; i < model->connections.size(); ++i) {
        Connection *conn = model->connections[i];
        this->instructions.push_back(
            new Instruction<int>(conn,
                (int*) this->state->output,
                this->state->input,
                this->state->get_matrix(conn->id)));
    }
}

void IzhikevichDriver::step_connections() {
    for (int i = 0; i < this->instructions.size(); ++i) {
        Instruction<int> *inst = this->instructions[i];
        // Determine which part of spike vector to use based on delay
        int word_index = HISTORY_SIZE - (inst->delay / 32) - 1;
        int mask = 1 << (inst->delay % 32);

        step<int, int>(inst, this->calc_input_ptr, mask);
    }
}

void IzhikevichDriver::step_state() {
    int num_neurons = this->state->num_neurons;

#ifdef PARALLEL
    int threads = 128;
    int blocks = calc_blocks(num_neurons, threads);
    izhikevich<<<blocks, threads>>>(
        this->iz_state->voltage,
        this->iz_state->recovery,
        this->iz_state->input,
        this->iz_state->neuron_parameters,
        num_neurons);
    cudaCheckError("Failed to update neuron voltages!");
    calc_spikes<<<blocks, threads>>>(
        this->iz_state->spikes,
        this->iz_state->voltage,
        this->iz_state->recovery,
        this->iz_state->neuron_parameters,
        num_neurons);
    cudaCheckError("Failed to timestep spikes!");
#else
    izhikevich(
        this->iz_state->voltage,
        this->iz_state->recovery,
        this->iz_state->input,
        this->iz_state->neuron_parameters,
        num_neurons);
    calc_spikes(
        this->iz_state->spikes,
        this->iz_state->voltage,
        this->iz_state->recovery,
        this->iz_state->neuron_parameters,
        num_neurons);
#endif
}

void IzhikevichDriver::step_weights() {
    /* 5. Update weights */
#ifdef PARALLEL
    cudaCheckError("Failed to update connection weights!");
#endif
}


/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
GLOBAL void izhikevich(float* voltages, float* recoveries,
        float* currents, IzhikevichParameters* neuron_params, int num_neurons) {
    float voltage, recovery, current, delta_v;
    IzhikevichParameters *params;

    /* 3. Voltage Updates */
    // For each neuron...
    //   update voltage according to current
    //     Euler's method with #defined resolution
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < num_neurons) {
#else
    for (int nid = 0 ; nid < num_neurons; ++nid) {
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
GLOBAL void calc_spikes(int* spikes, float* voltages,
        float* recoveries, IzhikevichParameters* neuron_params, int num_neurons) {
    int spike, new_value; 
    IzhikevichParameters *params;

    /* 4. Timestep */
    // Determine spikes.
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < num_neurons) {
#else
    for (int nid = 0; nid < num_neurons; ++nid) {
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
