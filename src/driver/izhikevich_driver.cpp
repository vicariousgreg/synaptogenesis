#include "driver/izhikevich_driver.h"

/* Voltage threshold for neuron spiking. */
#define SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define EULER_RES 10

/*****************************************************************************/
/************************* GENERIC IMPLEMENTATIONS ***************************/
/*****************************************************************************/

void IzhikevichDriver::step_connection_fully_connected(Connection *conn) {
    // Determine which part of spike vector to use based on delay
    int word_index = HISTORY_SIZE - (conn->delay / 32) - 1;
    int mask = 1 << (conn->delay % 32);

#ifdef PARALLEL
    int *spikes = &this->iz_state->device_spikes[this->model->num_neurons * word_index];
    int blocks = calc_blocks(conn->to_layer->size);
    calc_matrix<<<blocks, THREADS>>>(
        spikes + conn->from_layer->index,
        this->iz_state->get_matrix(conn->id),
        this->iz_state->device_input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        mask,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");

#else
    int *spikes = &this->iz_state->spikes[this->model->num_neurons * word_index];
    calc_matrix(
        spikes + conn->from_layer->index,
        this->state->get_matrix(conn->id),
        this->iz_state->input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        mask,
        conn->opcode);
#endif
}

void IzhikevichDriver::step_connection_one_to_one(Connection *conn) {
    // Determine which part of spike vector to use based on delay
    int word_index = HISTORY_SIZE - (conn->delay / 32) - 1;
    int mask = 1 << (conn->delay % 32);

#ifdef PARALLEL
    int *spikes = &this->iz_state->device_spikes[this->model->num_neurons * word_index];
    int blocks = calc_blocks(conn->to_layer->size);
    activate_vector<<<blocks, THREADS>>>(
        spikes + conn->from_layer->index,
        this->iz_state->get_matrix(conn->id),
        this->iz_state->device_input + conn->to_layer->index,
        conn->to_layer->size,
        mask,
        conn->opcode);
    cudaCheckError("Failed to calculate connection activation!");

#else
    int *spikes = &this->iz_state->spikes[this->model->num_neurons * word_index];
    activate_vector(
        spikes + conn->from_layer->index,
        this->iz_state->get_matrix(conn->id),
        this->iz_state->input + conn->to_layer->index,
        conn->to_layer->size,
        mask,
        conn->opcode);
#endif
}

void IzhikevichDriver::step_connection_divergent(Connection *conn, bool convolutional) {
    // Determine which part of spike vector to use based on delay
    int word_index = HISTORY_SIZE - (conn->delay / 32) - 1;
    int mask = 1 << (conn->delay % 32);

#ifdef PARALLEL
    int *spikes = &this->iz_state->device_spikes[this->model->num_neurons * word_index];
    dim3 blocks_per_grid(
        calc_blocks(conn->to_layer->rows, 1),
        calc_blocks(conn->to_layer->columns, 128));
    dim3 threads_per_block(1, 128);
    calc_matrix_divergent<<<blocks_per_grid, threads_per_block>>>(
        spikes + conn->from_layer->index,
        this->iz_state->get_matrix(conn->id),
        this->iz_state->device_input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        mask,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
    cudaCheckError("Failed to calculate connection activation!");

#else
    int *spikes = &this->iz_state->spikes[this->model->num_neurons * word_index];
    calc_matrix_divergent(
        spikes + conn->from_layer->index,
        this->state->get_matrix(conn->id),
        this->iz_state->input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        mask,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
#endif
}

void IzhikevichDriver::step_connection_convergent(Connection *conn, bool convolutional) {
    // Determine which part of spike vector to use based on delay
    int word_index = HISTORY_SIZE - (conn->delay / 32) - 1;
    int mask = 1 << (conn->delay % 32);

#ifdef PARALLEL
    int *spikes = &this->iz_state->device_spikes[this->model->num_neurons * word_index];
    dim3 blocks_per_grid(
        calc_blocks(conn->to_layer->rows, 1),
        calc_blocks(conn->to_layer->columns, 128));
    dim3 threads_per_block(1, 128);
    calc_matrix_convergent<<<blocks_per_grid, threads_per_block>>>(
        spikes + conn->from_layer->index,
        this->iz_state->get_matrix(conn->id),
        this->iz_state->device_input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        mask,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
    cudaCheckError("Failed to calculate connection activation!");

#else
    int *spikes = &this->iz_state->spikes[this->model->num_neurons * word_index];
    calc_matrix_convergent(
        spikes + conn->from_layer->index,
        this->state->get_matrix(conn->id),
        this->iz_state->input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        mask,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional);
#endif
}

void IzhikevichDriver::step_output() {
    int num_neurons = this->model->num_neurons;

#ifdef PARALLEL
    int blocks = calc_blocks(num_neurons);
    izhikevich<<<blocks, THREADS>>>(
        this->iz_state->device_voltage,
        this->iz_state->device_recovery,
        this->iz_state->device_input,
        this->iz_state->device_neuron_parameters,
        num_neurons);
    cudaCheckError("Failed to update neuron voltages!");
    calc_spikes<<<blocks, THREADS>>>(
        this->iz_state->device_spikes,
        this->iz_state->device_voltage,
        this->iz_state->device_recovery,
        this->iz_state->device_neuron_parameters,
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


#ifdef PARALLEL
/*****************************************************************************/
/************************ PARALLEL IMPLEMENTATIONS ***************************/
/*****************************************************************************/

KERNEL void calc_matrix(int* spikes, float* weights,
        float* currents, int from_size, int to_size, int mask, Opcode opcode) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < to_size) {
        float sum = 0;
        for (int row = 0 ; row < from_size ; ++row) {
            sum += (spikes[row] & mask) * weights[row * to_size + col];
        }
        currents[col] = calc(opcode, currents[col], sum);
    }
}

KERNEL void calc_matrix_divergent(int* spikes, float* weights,
        float* currents, int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional) {
    /*
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*to_columns + d_col;

    if (d_row < to_rows and d_col < to_columns) {
        int kernel_size = overlap * overlap;
        int kernel_row_size = (convolutional) ? 1 : from_rows * from_columns;

        float sum = 0.0;
        int d_index = d_row*to_columns + d_col;

        int start_s_row = d_row / overlap;
        int start_s_col = d_col / overlap ;
        int end_s_row = (d_row + stride) / overlap;
        int end_s_col = (d_col + stride) / overlap ;

        for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
            for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                // Convolutional connections share weights, and don't use an offset
                int s_index = (s_row * from_columns) + s_col;
                int kernel_offset = (convolutional) ? 0 : s_index * kernel_size;

                int k_row = (d_row + ((overlap - stride) * s_row) % overlap);
                int k_col = (d_col + ((overlap - stride) * s_col) % overlap);
                int k_index = kernel_offset + (k_row * overlap) + k_col;

                sum += (spikes[s_index] & mask) * weights[k_index];
            }
        }
        currents[d_index] = calc(opcode, currents[d_index], sum);
    }
    */
}

KERNEL void calc_matrix_convergent(int* spikes, float* weights,
        float* currents, int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*to_columns + d_col;

    if (d_row < to_rows and d_col < to_columns) {
        int kernel_size = overlap * overlap;
        int kernel_row_size = (convolutional) ? 1 : to_rows * to_columns;

        float sum = 0.0;
        int s_row = d_row * stride;
        int s_col = d_col * stride;

        // Convolutional connections share weights, and don't use an offset
        // In parallel version, matrix is transposed, so the offset is the index.
        int weight_col = (convolutional) ? 0 : d_index;

        // Run the kernel
        for (int k_row = 0 ; k_row < overlap ; ++k_row) {
            for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                int s_index = ((s_row+k_row) * from_columns) + (s_col+k_col);
                int weight_offset = (((k_row*overlap) + k_col) * kernel_row_size);
                sum += (spikes[s_index] & mask) * weights[weight_offset + weight_col];
            }
        }
        currents[d_index] = calc(opcode, currents[d_index], sum);
    }
}

KERNEL void activate_vector(int* spikes, float* weights,
            float* currents, int size, int mask, Opcode opcode) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        currents[index] = calc(opcode, currents[index],
            (spikes[index] & mask) * weights[index]);
    }
}

/* Parallel implementation of Izhikevich voltage update function.
 * Each thread calculates for one neuron.  Because this is a single
 *   dimensional calculation, few optimizations are possible. */
KERNEL void izhikevich(float* voltages, float* recoveries,
        float* currents, IzhikevichParameters* neuron_params, int num_neurons) {
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
KERNEL void calc_spikes(int* spikes, float* voltages,
        float* recoveries, IzhikevichParameters* neuron_params, int num_neurons) {
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
            IzhikevichParameters *params = &neuron_params[nid];
            voltages[nid] = params->c;
            recoveries[nid] += params->d;
        }
    }
}

#else
/*****************************************************************************/
/************************** SERIAL IMPLEMENTATIONS ***************************/
/*****************************************************************************/

void calc_matrix(int* spikes, float* weights, float* currents,
                          int from_size, int to_size, int mask, Opcode opcode) {
    // IMPORTANT:
    // Serial implementation is faster if matrix is interpreted in a transposed
    //    fashion compared to parallel.  In this loop, row is the destination,
    //    column is the source.  In this way, inputs to one neuron are
    //    contiguous in memory.
    for (int row = 0 ; row < to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < from_size ; ++col) {
            sum += (spikes[col] & mask) * weights[row*from_size + col];
        }
        currents[row] = calc(opcode, currents[row], sum);
    }
}

void calc_matrix_divergent(int* spikes, float* weights, float* currents,
        int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional) {
    int kernel_size = overlap * overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
            float sum = 0.0;
            int d_index = d_row*to_columns + d_col;

            int start_s_row = d_row / overlap;
            int start_s_col = d_col / overlap ;
            int end_s_row = (d_row + stride) / overlap;
            int end_s_col = (d_col + stride) / overlap ;

            for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                    // Convolutional connections share weights, and don't use an offset
                    int s_index = (s_row * from_columns) + s_col;
                    int weight_offset = (convolutional) ? 0 : s_index * kernel_size;

                    int k_row = (d_row + ((overlap - stride) * s_row) % overlap);
                    int k_col = (d_col + ((overlap - stride) * s_col) % overlap);
                    int k_index = (k_row * overlap) + k_col;

                    sum += (spikes[s_index] & mask) * weights[k_index + weight_offset];
                }
            }
            currents[d_index] = calc(opcode, currents[d_index], sum);
        }
    }
}

void calc_matrix_convergent(int* spikes, float* weights, float* currents,
        int from_rows, int from_columns, int to_rows, int to_columns,
        int mask, Opcode opcode, int overlap, int stride, bool convolutional) {
    int kernel_size = overlap * overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
            float sum = 0.0;
            int s_row = d_row * stride;
            int s_col = d_col * stride;
            int d_index = d_row*to_columns + d_col;

            // Convolutional connections share weights, and don't use an offset
            int weight_offset = (convolutional) ? 0 : d_index * kernel_size;

            // Run the kernel (unshared)
            for (int k_row = 0 ; k_row < overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                    int s_index = ((s_row+k_row) * from_columns) + (s_col+k_col);
                    int weight_col = (k_row*overlap) + k_col;
                    sum += (spikes[s_index] & mask) *
                        weights[weight_offset + weight_col];
                }
            }
            currents[d_index] = calc(opcode, currents[d_index], sum);
        }
    }
}

void activate_vector(int* spikes, float* weights,
                        float* currents, int size, int mask, Opcode opcode) {
    for (int index = 0 ; index < size ; ++index) {
        currents[index] = calc(opcode, currents[index],
            (spikes[index] & mask) * weights[index]);
    }
}

void izhikevich(float* voltages, float* recoveries, float* currents,
                IzhikevichParameters* neuron_params, int num_neurons) {
    float voltage, recovery, current, delta_v;
    IzhikevichParameters *params;

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
                 IzhikevichParameters* neuron_params, int num_neurons) {
    int spike, new_value; 
    IzhikevichParameters *params;

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
