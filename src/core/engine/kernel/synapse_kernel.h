#ifndef synapse_kernel_h
#define synapse_kernel_h

#include "engine/kernel/synapse_data.h"
#include "engine/kernel/extractor.h"
#include "util/parallel.h"
#include "util/constants.h"
#include "util/tools.h"
#include "util/pointer.h"

// Different min and max functions are used on the host and device
#ifdef __CUDACC__
#define MIN min
#define MAX max
#else
#include <algorithm>
#define MIN std::fmin
#define MAX std::fmax
#endif

/* Typedef for kernel functions, which just take SynapseData */
typedef SynapseData SYNAPSE_ARGS;
typedef void(*SYNAPSE_KERNEL)(SYNAPSE_ARGS);

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
        case POOL: return MAX(prior, (1+input));
    }
    return 0.0;
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


// Extract fields from synapse_data
// This makes a surprising difference in runtime
// This macro only contains extractions relevant to all connection kernels
#define SYNAPSE_PREAMBLE \
    const Opcode opcode = synapse_data.opcode; \
    const int delay = synapse_data.delay; \
    float * const weights = synapse_data.weights.get(); \
    const int num_weights = synapse_data.num_weights; \
    const bool plastic = synapse_data.plastic; \
    const float max_weight = synapse_data.max_weight; \
    const int from_size = synapse_data.from_size; \
    const int from_rows = synapse_data.from_rows; \
    const int from_columns = synapse_data.from_columns; \
    const int to_size = synapse_data.to_size; \
    const int to_rows = synapse_data.to_rows; \
    const int to_columns = synapse_data.to_columns; \
    Output * const outputs = synapse_data.outputs.get(); \
    Output * const destination_outputs = synapse_data.destination_outputs.get(); \
    float * const inputs = synapse_data.inputs.get(); \
    const EXTRACTOR extractor = synapse_data.extractor;

#define FULLY_CONNECTED_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    EXTRACTIONS; \
 \
    for (int to_index = 0 ; to_index < to_size ; ++to_index) { \
        NEURON_PRE; \
        for (int from_index = 0 ; from_index < from_size ; ++from_index) { \
            int weight_index = to_index * from_size + from_index; \
            WEIGHT_OP; \
        } \
        NEURON_POST; \
    } \
}


#define FULLY_CONNECTED_PARALLEL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    EXTRACTIONS; \
 \
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < to_size) { \
        NEURON_PRE; \
        for (int from_index = 0 ; from_index < from_size ; ++from_index) { \
            int weight_index = from_index * to_size + to_index; \
            WEIGHT_OP; \
        } \
        NEURON_POST; \
    } \
}



#define ONE_TO_ONE_SERIAL(FUNC_NAME, EXTRACTIONS, WEIGHT_OP) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    EXTRACTIONS; \
 \
    for (int index = 0 ; index < to_size ; ++index) { \
        WEIGHT_OP; \
    } \
}

#define ONE_TO_ONE_PARALLEL(FUNC_NAME, EXTRACTIONS, WEIGHT_OP) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    EXTRACTIONS; \
 \
    int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index < to_size) { \
        WEIGHT_OP; \
    } \
}



#define CONVERGENT_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    const bool convolutional = synapse_data.convolutional; \
    const int field_size = synapse_data.field_size; \
    const int stride = synapse_data.stride; \
    const int fray = synapse_data.fray; \
    EXTRACTIONS; \
 \
    int kernel_size = field_size * field_size; \
 \
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) { \
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) { \
            int to_index = d_row * to_columns + d_col; \
            NEURON_PRE; \
 \
            /* Determine starting row and column for source neurons */ \
            int s_row = d_row * stride - fray; \
            int s_col = d_col * stride - fray; \
 \
            /* Row of matrix is either the first column (convolutional) */ \
            /*   or the index of the destination neuron otherwise */ \
            int weight_offset = (convolutional) ? 0 : to_index * kernel_size; \
 \
            /* Run the kernel */ \
            for (int k_row = 0 ; k_row < field_size ; ++k_row) { \
                for (int k_col = 0 ; k_col < field_size ; ++k_col) { \
                    int k_s_row = s_row + k_row; \
                    int k_s_col = s_col + k_col; \
 \
                    /* The connection is frayed if the layers are the same size */ \
                    /* Avoid making connections with non-existent neurons! */ \
                    if (fray != 0 and (k_s_row < 0 or k_s_row >= from_rows \
                        or k_s_col < 0 or k_s_col >= from_columns)) \
                        continue; \
 \
                    int from_index = k_s_row * from_columns + k_s_col; \
 \
                    /* Column of matrix is the kernel index */ \
                    int weight_index = weight_offset + \
                        (k_row * field_size) + k_col; \
 \
                    WEIGHT_OP; \
                } \
            } \
            NEURON_POST; \
        } \
    } \
}

#define CONVERGENT_PARALLEL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    const bool convolutional = synapse_data.convolutional; \
    const int field_size = synapse_data.field_size; \
    const int stride = synapse_data.stride; \
    const int fray = synapse_data.fray; \
    EXTRACTIONS; \
 \
    int kernel_size = field_size * field_size; \
 \
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < to_size) { \
        int d_row = to_index / to_columns; \
        int d_col = to_index % to_columns; \
        NEURON_PRE; \
\
        /* Determine starting row and column for source neurons */ \
        int s_row = d_row * stride - fray; \
        int s_col = d_col * stride - fray; \
\
        /* Column of matrix is either the first column (convolutional) */ \
        /*   or the index of the destination neuron otherwise */ \
        int weight_col = (convolutional) ? 0 : to_index; \
        /* Kernels are organized into columns */ \
        /* One kernel per destination neuron */ \
        /*   Unless convolutional (shared kernel) */ \
        int kernel_row_size = (convolutional) ? 1 : to_size; \
\
        /* Run the kernel */ \
        for (int k_row = 0 ; k_row < field_size ; ++k_row) { \
            for (int k_col = 0 ; k_col < field_size ; ++k_col) { \
                int k_s_row = s_row + k_row; \
                int k_s_col = s_col + k_col; \
\
                /* The connection is frayed if the layers are the same size */ \
                /* Avoid making connections with non-existent neurons! */ \
                if (fray != 0 and (k_s_row < 0 or k_s_row >= from_rows \
                    or k_s_col < 0 or k_s_col >= from_columns)) \
                    continue; \
\
                int from_index = k_s_row * from_columns + k_s_col; \
\
                /* Row of matrix is the kernel index * row size (see above) */ \
                int weight_index = weight_col + \
                    ((k_row*field_size) + k_col) * kernel_row_size; \
\
                WEIGHT_OP; \
            } \
        } \
        NEURON_POST; \
    } \
}

#define DIVERGENT_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    const int field_size = synapse_data.field_size; \
    const int stride = synapse_data.stride; \
    const int fray = synapse_data.fray; \
    const int kernel_size = field_size * field_size; \
    EXTRACTIONS; \
\
    /* Iterate over destination neurons */ \
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) { \
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) { \
            int to_index = d_row*to_columns + d_col; \
			NEURON_PRE; \
\
            /* Determine range of source neurons for divergent kernel */ \
            int start_s_row = (d_row + fray - field_size + stride) / stride; \
            int start_s_col = (d_col + fray - field_size + stride) / stride; \
            int end_s_row = (d_row + fray) / stride; \
            int end_s_col = (d_col + fray) / stride; \
\
            int start = field_size - stride; \
            int offset = (stride > (field_size/2)) \
                         ? (2*stride - field_size) : (stride-1);\
\
            /* Iterate over relevant source neurons... */ \
            int k_row = start + ((d_row + fray + offset) % stride); \
            for (int s_row = start_s_row ; s_row <= end_s_row ; (++s_row, k_row -= stride)) { \
                int k_col = start + ((d_col + fray + offset) % stride); \
                for (int s_col = start_s_col ; s_col <= end_s_col ; (++s_col, k_col -= stride)) { \
                    /* Avoid making connections with non-existent neurons! */ \
                    if (fray != 0 and (s_row < 0 or s_row >= from_rows \
                        or s_col < 0 or s_col >= from_columns)) \
                        continue; \
\
                    int from_index = (s_row * from_columns) + s_col; \
					int weight_index = (from_index * kernel_size) \
						+ (k_row * field_size) + k_col; \
					WEIGHT_OP; \
                } \
            } \
			NEURON_POST; \
        } \
    } \
}

#define DIVERGENT_PARALLEL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    const int field_size = synapse_data.field_size; \
    const int stride = synapse_data.stride; \
    const int fray = synapse_data.fray; \
    EXTRACTIONS; \
\
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < to_size) { \
        int d_row = to_index / to_columns; \
        int d_col = to_index % to_columns; \
        NEURON_PRE; \
 \
        /* Determine range of source neurons for divergent kernel */ \
        int start_s_row = (d_row + fray - field_size + stride) / stride; \
        int start_s_col = (d_col + fray - field_size + stride) / stride; \
        int end_s_row = (d_row + fray) / stride; \
        int end_s_col = (d_col + fray) / stride; \
\
        int start = field_size - stride; \
        int offset = (stride > (field_size/2)) \
                     ? (2*stride - field_size) : (stride-1);\
\
        /* Kernels are organized into columns
           One kernel per source neuron */ \
        int kernel_size = field_size * field_size; \
        int kernel_row_size = from_rows * from_columns; \
\
        /* Iterate over relevant source neurons... */ \
        int k_row = start + ((d_row + fray + offset) % stride); \
        for (int s_row = start_s_row ; s_row <= end_s_row ; (++s_row, k_row -= stride)) { \
            int k_col = start + ((d_col + fray + offset) % stride); \
            for (int s_col = start_s_col ; s_col <= end_s_col ; (++s_col, k_col -= stride)) { \
                /* Avoid making connections with non-existent neurons! */ \
                if (fray != 0 and (s_row < 0 or s_row >= from_rows \
                    or s_col < 0 or s_col >= from_columns)) \
                    continue; \
\
                int from_index = (s_row * from_columns) + s_col; \
\
                /* Row of matrix is the kernel index * row size (see above)
                   Column of matrix is the index of the source neuron */ \
                int weight_index = from_index + \
					(((k_row * field_size) + k_col) * kernel_row_size); \
                 WEIGHT_OP; \
            } \
        } \
        NEURON_POST; \
    } \
}

#ifdef __CUDACC__

#define CALC_FULLY_CONNECTED(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
FULLY_CONNECTED_PARALLEL(FUNC_NAME##_PARALLEL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
FULLY_CONNECTED_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() {\
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#define CALC_ONE_TO_ONE(FUNC_NAME, EXTRACTIONS, WEIGHT_OP) \
ONE_TO_ONE_PARALLEL(FUNC_NAME##_PARALLEL, EXTRACTIONS, WEIGHT_OP) \
ONE_TO_ONE_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, WEIGHT_OP) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#define CALC_CONVERGENT(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
CONVERGENT_PARALLEL(FUNC_NAME##_PARALLEL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
CONVERGENT_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#define CALC_DIVERGENT(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DIVERGENT_PARALLEL(FUNC_NAME##_PARALLEL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DIVERGENT_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}


#else


#define CALC_FULLY_CONNECTED(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
FULLY_CONNECTED_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL); \
}

#define CALC_ONE_TO_ONE(FUNC_NAME, EXTRACTIONS, WEIGHT_OP) \
ONE_TO_ONE_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, WEIGHT_OP) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL); \
}

#define CALC_CONVERGENT(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
CONVERGENT_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL); \
}

#define CALC_DIVERGENT(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DIVERGENT_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL); \
}

#endif

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

#define CALC_VAL(from_index, weight_index) \
    float val = extractor(outputs[from_index], delay) * weights[weight_index];

#define AGGREGATE(to_index, sum) \
    inputs[to_index] = calc(opcode, inputs[to_index], sum);

#define ACTIVATE_FULLY_CONNECTED(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_FULLY_CONNECTED(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum */ \
    CALC_VAL(from_index, weight_index); \
    sum += val; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * Aggregate sum to input */ \
    AGGREGATE(to_index, sum); \
)

#define ACTIVATE_ONE_TO_ONE(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_ONE_TO_ONE(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum
     * Aggregate weight input to total input */ \
    CALC_VAL(index, index); \
    UPDATE_CALC; \
    AGGREGATE(index, val); \
)

#define ACTIVATE_CONVERGENT(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_CONVERGENT(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum */ \
    CALC_VAL(from_index, weight_index); \
    sum += val; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * Aggregate sum to input */ \
    AGGREGATE(to_index, sum); \
)

#define ACTIVATE_DIVERGENT(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_DIVERGENT(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
 \
    /* WEIGHT_OP
     * Calculate weight input, add to sum */ \
    CALC_VAL(from_index, weight_index); \
    sum += val; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * Aggregate sum to input */ \
    AGGREGATE(to_index, sum); \
)

#endif
