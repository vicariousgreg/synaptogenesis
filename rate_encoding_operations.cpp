#include <math.h>

#include "rate_encoding_operations.h"
#include "constants.h"


/*****************************************************************************/
/************************* GENERIC IMPLEMENTATIONS ***************************/
/*****************************************************************************/
void re_update_inputs(Connection &conn, float* mData, void* outputs,
                     float* inputs, int num_neurons) {
#ifdef PARALLEL
    int threads = 32;
    int blocks = ceil((float)(conn.to_layer.size) / threads);
#endif

    if (conn.type == FULLY_CONNECTED) {
#ifdef PARALLEL
        parallel_activate_matrix<<<blocks, threads>>>(
#else
        serial_activate_matrix(
#endif
            (float*)outputs + conn.from_layer.index,
            mData,
            inputs + conn.to_layer.index,
            conn.from_layer.size,
            conn.to_layer.size,
            conn.opcode);
    } else if (conn.type == ONE_TO_ONE) {
#ifdef PARALLEL
        parallel_activate_vector<<<blocks, threads>>>(
#else
        serial_activate_vector(
#endif
            (float*)outputs + conn.from_layer.index,
            mData,
            inputs + conn.to_layer.index,
            conn.to_layer.size,
            conn.opcode);
    }
#ifdef PARALLEL
    cudaCheckError("Failed to calculate connection activation!");
#endif
}

void re_update_outputs(void* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons) {
#ifdef PARALLEL
    int threads = 32;
    int blocks = ceil((float)(num_neurons) / threads);
    parallel_activation_function<<<blocks, threads>>>(
#else
    serial_activation_function(
#endif
        (float*)outputs,
        inputs,
        neuron_params,
        num_neurons);

#ifdef PARALLEL
    cudaCheckError("Failed to update neuron output!");
#endif
}

void re_update_weights() {
#ifdef PARALLEL
    cudaCheckError("Failed to update connection weights!");
#endif
}


#ifdef PARALLEL
/*****************************************************************************/
/************************ PARALLEL IMPLEMENTATIONS ***************************/
/*****************************************************************************/

__global__ void parallel_activate_matrix(float* outputs, float* weights,
        float* inputs, int from_size, int to_size, OPCODE opcode) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < to_size) {
        float sum = 0;
        for (int row = 0 ; row < from_size ; ++row) {
            sum += outputs[row] * weights[row * to_size + col];
        }
        inputs[col] = calc(opcode, inputs[col], sum);
    }
}

__global__ void parallel_activate_vector(float* outputs, float* weights,
                    float* inputs, int size, OPCODE opcode) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        inputs[index] = calc(opcode, inputs[index],
            outputs[index] * weights[index]);
    }
}

__global__ void parallel_activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < num_neurons and inputs[nid] > 0.0) {
        outputs[nid] = tanh(0.1*inputs[nid]);
    }
}

#else
/*****************************************************************************/
/************************** SERIAL IMPLEMENTATIONS ***************************/
/*****************************************************************************/

void serial_activate_matrix(float* outputs, float* weights, float* inputs,
                        int from_size, int to_size, OPCODE opcode) {
    // IMPORTANT:
    // Serial implementation is faster if matrix is interpreted in a transposed
    //    fashion compared to parallel.  In this loop, row is the destination,
    //    column is the source.  In this way, inputs to one neuron are
    //    contiguous in memory.
    for (int row = 0 ; row < to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < from_size ; ++col) {
            sum += outputs[col] * weights[row*from_size + col];
        }
        inputs[row] = calc(opcode, inputs[row], sum);
    }
}

void serial_activate_vector(float* outputs, float* weights, float* inputs,
                                        int size, OPCODE opcode) {
    for (int index = 0 ; index < size ; ++index) {
        inputs[index] = calc(opcode, inputs[index],
            outputs[index] * weights[index]);
    }
}

void serial_activation_function(float* outputs, float* inputs,
                RateEncodingParameters* neuron_params, int num_neurons) {
    RateEncodingParameters *params;

    for (int nid = 0 ; nid < num_neurons; ++nid) {
        if (inputs[nid] > 0.0) {
            outputs[nid] = tanh(0.1*inputs[nid]);
        }
    }
}

#endif
