#ifndef kernel_h
#define kernel_h

#include "engine/kernel/kernel_data.h"
#include "util/parallel.h"
#include "util/constants.h"

#define MOD_RATE 0.3
#define MOD_DECAY 0.01
#define MOD_MAX 10.0
#define SUM_COEFFICIENT 0.5
#define WEIGHT_DECAY 0.025

/* Typedef for kernel functions, which just take KernelData */
typedef void(*KERNEL)(KernelData);

/* Synaptic operations
 * |prior| is the current state of the neuron.
 * |input| is the synaptic input accomulated from one connection.
 *
 * ADD represent traditional excitatory input
 * SUB represent traditional inhibitory input
 * MULT and DIV represent modulatory input that can be used for gating
 * */
inline DEVICE float calc(Opcode opcode, float prior, float input) {
    switch (opcode) {
        case ADD:  return prior + input;
        case SUB:  return prior - input;
        case MULT: return prior * (1+input);
        case DIV:  return prior / (1+input);
    }
    return 0.0;
}

/* Clears input data */
inline GLOBAL void clear_data(float* data, int count) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
#else
    for (int nid = 0; nid < count; ++nid)
#endif
        data[nid] = 0.0;
}

/******************************************************************************/
/*************************** CONNECTION KERNELS *******************************/
/******************************************************************************/

/* Parallel and Serial kernel macros for connection functions.
 * The Parallel versions determine an index based on the CUDA thread/block.
 * The Serial versions perform iterations over all of the neurons.
 *
 * IMPORTANT:
 * Serial implementation is faster if matrix is interpreted in a transposed
 *    fashion compared to parallel.  For serial loops, row is the destination,
 *    column is the source.  This way, inputs to one neuron are contiguous in
 *    memory.  For the purposes of these macros, "row" and "col" are avoided
 *    in favor of "to_index" and "from_index" so that the macros can be used
 *    without the necessity of knowing which implementation is used.
 *
 * Macros take the following arguments:
 *   - FUNC_NAME: name of the resulting function
 *   - EXTRACTIONS: variables extracted from kernel data at beginning of function
 *        anything necessary for other arguments should be extracted here
 *   - NEURON_PRE: operation performed at beginning of loop per neuron
 *   - WEIGHT_OP: operation performed on each weight
 *   - NEURON_POST: operation performed at end of loop per neuron
 *
 * These macros can be used to define functions that operate on connections.
 * So far, this includes:
 *     - connection activation (input calculations)
 *     - weight updates (learning rules)
 */

#define FULLY_CONNECTED_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(KernelData kernel_data) { \
    EXTRACTIONS; \
 \
    for (int to_index = 0 ; to_index < kernel_data.to_size ; ++to_index) { \
        NEURON_PRE; \
        for (int from_index = 0 ; from_index < kernel_data.from_size ; ++from_index) { \
            int weight_index = to_index * kernel_data.from_size + from_index; \
            WEIGHT_OP; \
        } \
        NEURON_POST; \
    } \
}


#define FULLY_CONNECTED_PARALLEL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(KernelData kernel_data) { \
    EXTRACTIONS; \
 \
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < kernel_data.to_size) { \
        NEURON_PRE; \
        for (int from_index = 0 ; from_index < kernel_data.from_size ; ++from_index) { \
            int weight_index = from_index * kernel_data.to_size + to_index; \
            WEIGHT_OP; \
        } \
        NEURON_POST; \
    } \
}



#define ONE_TO_ONE_SERIAL(FUNC_NAME, EXTRACTIONS, WEIGHT_OP) \
GLOBAL void FUNC_NAME(KernelData kernel_data) { \
    EXTRACTIONS; \
 \
    for (int index = 0 ; index < kernel_data.to_size ; ++index) { \
        WEIGHT_OP; \
    } \
}

#define ONE_TO_ONE_PARALLEL(FUNC_NAME, EXTRACTIONS, WEIGHT_OP) \
GLOBAL void FUNC_NAME(KernelData kernel_data) { \
    EXTRACTIONS; \
 \
    int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index < kernel_data.to_size) { \
        WEIGHT_OP; \
    } \
}



#define CONVERGENT_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(KernelData kernel_data) { \
    EXTRACTIONS; \
 \
    int kernel_size = kernel_data.overlap * kernel_data.overlap; \
 \
    for (int d_row = 0 ; d_row < kernel_data.to_rows ; ++d_row) { \
        for (int d_col = 0 ; d_col < kernel_data.to_columns ; ++d_col) { \
            int to_index = d_row * kernel_data.to_columns + d_col; \
            NEURON_PRE; \
 \
            /* Determine starting row and column for source neurons */ \
            int s_row = d_row * kernel_data.stride - kernel_data.fray; \
            int s_col = d_col * kernel_data.stride - kernel_data.fray; \
 \
            /* Row of matrix is either the first column (convolutional) */ \
            /*   or the index of the destination neuron otherwise */ \
            int weight_offset = (kernel_data.convolutional) \
                                ? 0 : to_index * kernel_size; \
 \
            /* Run the kernel */ \
            for (int k_row = 0 ; k_row < kernel_data.overlap ; ++k_row) { \
                for (int k_col = 0 ; k_col < kernel_data.overlap ; ++k_col) { \
                    int k_s_row = s_row + k_row; \
                    int k_s_col = s_col + k_col; \
 \
                    /* The connection is frayed if the layers are the same size */ \
                    /* Avoid making connections with non-existent neurons! */ \
                    if (kernel_data.fray != 0 and (k_s_row < 0 or k_s_row >= kernel_data.to_rows \
                        or k_s_col < 0 or k_s_col >= kernel_data.to_columns)) \
                        continue; \
 \
                    int from_index = k_s_row * kernel_data.from_columns + k_s_col; \
 \
                    /* Column of matrix is the kernel index */ \
                    int weight_col = (k_row * kernel_data.overlap) + k_col; \
                    int weight_index = weight_offset + weight_col; \
 \
                    WEIGHT_OP; \
                } \
            } \
            NEURON_POST; \
        } \
    } \
}

#define CONVERGENT_PARALLEL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(KernelData kernel_data) { \
    EXTRACTIONS; \
 \
    int kernel_size = kernel_data.overlap * kernel_data.overlap; \
 \
    int d_row = blockIdx.x * blockDim.x + threadIdx.x; \
    int d_col = blockIdx.y * blockDim.y + threadIdx.y; \
    if (d_row < kernel_data.to_rows and d_col < kernel_data.to_columns) { \
        int to_index = d_row * kernel_data.to_columns + d_col; \
        NEURON_PRE; \
\
        /* Determine starting row and column for source neurons */ \
        int s_row = d_row * kernel_data.stride - kernel_data.fray; \
        int s_col = d_col * kernel_data.stride - kernel_data.fray; \
\
        /* Column of matrix is either the first column (convolutional) */ \
        /*   or the index of the destination neuron otherwise */ \
        int weight_col = (kernel_data.convolutional) ? 0 : to_index; \
        /* Kernels are organized into columns */ \
        /* One kernel per destination neuron */ \
        /*   Unless convolutional (shared kernel) */ \
        int kernel_row_size = (kernel_data.convolutional) \
                              ? 1 : kernel_data.to_size; \
\
        /* Run the kernel */ \
        for (int k_row = 0 ; k_row < kernel_data.overlap ; ++k_row) { \
            for (int k_col = 0 ; k_col < kernel_data.overlap ; ++k_col) { \
                int k_s_row = s_row + k_row; \
                int k_s_col = s_col + k_col; \
\
                /* The connection is frayed if the layers are the same size */ \
                /* Avoid making connections with non-existent neurons! */ \
                if (kernel_data.fray != 0 and (k_s_row < 0 or k_s_row >= kernel_data.to_rows \
                    or k_s_col < 0 or k_s_col >= kernel_data.to_columns)) \
                    continue; \
\
                int from_index = k_s_row * kernel_data.from_columns + k_s_col; \
\
                /* Row of matrix is the kernel index * row size (see above) */ \
                int weight_offset = \
                    ((k_row*kernel_data.overlap) + k_col) \
                    * kernel_row_size; \
                int weight_index = weight_offset + weight_col; \
\
                WEIGHT_OP; \
            } \
        } \
        NEURON_POST; \
    } \
}

#ifdef PARALLEL
#define CALC_FULLY_CONNECTED FULLY_CONNECTED_PARALLEL
#define CALC_ONE_TO_ONE ONE_TO_ONE_PARALLEL
#define CALC_CONVERGENT CONVERGENT_PARALLEL
#else
#define CALC_FULLY_CONNECTED FULLY_CONNECTED_SERIAL
#define CALC_ONE_TO_ONE ONE_TO_ONE_SERIAL
#define CALC_CONVERGENT CONVERGENT_SERIAL
#endif

#endif
