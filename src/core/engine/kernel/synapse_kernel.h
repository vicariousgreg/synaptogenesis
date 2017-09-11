#ifndef synapse_kernel_h
#define synapse_kernel_h

#include "engine/kernel/synapse_data.h"
#include "engine/kernel/extractor.h"
#include "util/parallel.h"
#include "util/constants.h"
#include "util/tools.h"
#include "util/pointer.h"

// Different min, max, and assert functions are used on the host and device
#ifdef __CUDACC__
#define MIN min
#define MAX max
#else
#include <algorithm>
#include <assert.h>
#define MIN std::fmin
#define MAX std::fmax
#endif

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
        // case MULT: return prior * (1+input);
        // case DIV:  return prior / (1+input);
        case MULT: return prior * input;
        case DIV:  return prior / input;
        case POOL: return MAX(prior, (1+input));
        default: assert(false);
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



#define SUBSET_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    const int from_row_start = synapse_data.subset_config.from_row_start; \
    const int from_row_end = synapse_data.subset_config.from_row_end; \
    const int from_col_start = synapse_data.subset_config.from_col_start; \
    const int from_col_end = synapse_data.subset_config.from_col_end; \
    const int to_row_start = synapse_data.subset_config.to_row_start; \
    const int to_row_end = synapse_data.subset_config.to_row_end; \
    const int to_col_start = synapse_data.subset_config.to_col_start; \
    const int to_col_end = synapse_data.subset_config.to_col_end; \
    const int from_kernel_size = synapse_data.subset_config.from_size; \
    EXTRACTIONS; \
 \
    int to_kernel_index = 0; \
    for (int to_row = to_row_start ; to_row < to_row_end ; ++to_row) { \
        for (int to_col = to_col_start ; to_col < to_col_end ; ++to_col) { \
            int to_index = to_row * to_columns + to_col; \
            NEURON_PRE; \
            int from_kernel_index = 0; \
\
            for (int from_row = from_row_start ; from_row < from_row_end ; ++from_row) { \
                for (int from_col = from_col_start ; from_col < from_col_end ; ++from_col) { \
                    int from_index = from_row * from_columns + from_col; \
                    int weight_index = to_kernel_index * from_kernel_size + from_kernel_index; \
\
                    WEIGHT_OP; \
\
                    ++from_kernel_index; \
                } \
            } \
\
            NEURON_POST; \
            ++to_kernel_index; \
        } \
    } \
}

#define SUBSET_PARALLEL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    const int from_row_start = synapse_data.subset_config.from_row_start; \
    const int from_row_end = synapse_data.subset_config.from_row_end; \
    const int from_col_start = synapse_data.subset_config.from_col_start; \
    const int from_col_end = synapse_data.subset_config.from_col_end; \
    const int to_row_start = synapse_data.subset_config.to_row_start; \
    const int to_row_size = synapse_data.subset_config.to_row_size; \
    const int to_col_start = synapse_data.subset_config.to_col_start; \
    const int to_col_size = synapse_data.subset_config.to_col_size; \
    const int to_kernel_size = synapse_data.subset_config.to_size; \
    EXTRACTIONS; \
 \
    int to_kernel_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_kernel_index < to_kernel_size) { \
        int to_row = (to_kernel_index / to_col_size) + to_row_start; \
        int to_col = (to_kernel_index % to_col_size) + to_col_start; \
        int to_index = to_row * to_columns + to_col; \
        int from_kernel_index = 0; \
        NEURON_PRE; \
\
        for (int from_row = from_row_start ; from_row < from_row_end ; ++from_row) { \
            for (int from_col = from_col_start ; from_col < from_col_end ; ++from_col) { \
                int from_index = from_row * from_columns + from_col; \
                int weight_index = from_kernel_index * to_kernel_size + to_kernel_index; \
\
                WEIGHT_OP; \
\
                ++from_kernel_index; \
            } \
        } \
\
        NEURON_POST; \
    } \
}



#define ONE_TO_ONE_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    EXTRACTIONS; \
 \
    for (int weight_index=0, from_index=0, to_index=0 ; \
         weight_index < from_size ; \
         ++weight_index, ++to_index, ++from_index) { \
        NEURON_PRE; \
        WEIGHT_OP; \
        NEURON_POST; \
    } \
}

#define ONE_TO_ONE_PARALLEL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    EXTRACTIONS; \
 \
    int weight_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (weight_index < from_size) { \
        int from_index = weight_index; \
        int to_index = weight_index; \
        NEURON_PRE; \
        WEIGHT_OP; \
        NEURON_POST; \
    } \
}



#define CONVERGENT_SERIAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
GLOBAL void FUNC_NAME(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    const bool convolutional = synapse_data.convolutional; \
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const int kernel_size = row_field_size * column_field_size; \
    const bool wrap = synapse_data.arborized_config.wrap; \
    EXTRACTIONS; \
 \
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) { \
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) { \
            int to_index = d_row * to_columns + d_col; \
            NEURON_PRE; \
 \
            /* Determine starting row and column for source neurons */ \
            int s_row = d_row * row_stride + row_offset; \
            int s_col = d_col * column_stride + column_offset; \
 \
            /* Row of matrix is either the first column (convolutional) */ \
            /*   or the index of the destination neuron otherwise */ \
            int weight_offset = (convolutional) ? 0 : to_index * kernel_size; \
 \
            /* Run the kernel */ \
            int k_index = 0; \
            for (int k_row = 0 ; k_row < row_field_size ; ++k_row) { \
                for (int k_col = 0 ; k_col < column_field_size ; (++k_col, ++k_index)) { \
                    int k_s_row = s_row + k_row; \
                    int k_s_col = s_col + k_col; \
 \
                    /* If wrapping, adjust out of bounds indices accordingly */ \
                    if (wrap) { \
                        k_s_row = (k_s_row < 0) \
                            ? k_s_row + from_rows \
                            : (k_s_row >= from_rows) \
                                ? k_s_row - from_rows : k_s_row; \
    \
                        k_s_col = (k_s_col < 0) \
                            ? k_s_col + from_columns \
                            : (k_s_col >= from_columns) \
                                ? k_s_col - from_columns : k_s_col; \
                    /* Avoid making connections with non-existent neurons */ \
                    } else if (k_s_row < 0 or k_s_row >= from_rows \
                        or k_s_col < 0 or k_s_col >= from_columns) { \
                        continue; \
                    } \
 \
                    int from_index = k_s_row * from_columns + k_s_col; \
 \
                    /* Column of matrix is the kernel index */ \
                    int weight_index = weight_offset + k_index; \
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
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const int kernel_size = row_field_size * column_field_size; \
    const bool wrap = synapse_data.arborized_config.wrap; \
    EXTRACTIONS; \
 \
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < to_size) { \
        int d_row = to_index / to_columns; \
        int d_col = to_index % to_columns; \
        NEURON_PRE; \
\
        /* Determine starting row and column for source neurons */ \
        int s_row = d_row * row_stride + row_offset; \
        int s_col = d_col * column_stride + column_offset; \
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
        int k_index = 0; \
        for (int k_row = 0 ; k_row < row_field_size ; ++k_row) { \
            for (int k_col = 0 ; k_col < column_field_size ; (++k_col, ++k_index)) { \
                int k_s_row = s_row + k_row; \
                int k_s_col = s_col + k_col; \
\
                /* If wrapping, adjust out of bounds indices accordingly */ \
                if (wrap) { \
                    k_s_row = (k_s_row < 0) \
                        ? k_s_row + from_rows \
                        : (k_s_row >= from_rows) \
                            ? k_s_row - from_rows : k_s_row; \
\
                    k_s_col = (k_s_col < 0) \
                        ? k_s_col + from_columns \
                        : (k_s_col >= from_columns) \
                            ? k_s_col - from_columns : k_s_col; \
                /* Avoid making connections with non-existent neurons */ \
                } else if (k_s_row < 0 or k_s_row >= from_rows \
                    or k_s_col < 0 or k_s_col >= from_columns) { \
                    continue; \
                } \
\
                int from_index = k_s_row * from_columns + k_s_col; \
\
                /* Row of matrix is the kernel index * row size (see above) */ \
                int weight_index = weight_col + k_index * kernel_row_size; \
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
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const int kernel_size = row_field_size * column_field_size; \
    const bool wrap = synapse_data.arborized_config.wrap; \
    EXTRACTIONS; \
\
    /* Iterate over destination neurons */ \
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) { \
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) { \
            int to_index = d_row*to_columns + d_col; \
            NEURON_PRE; \
\
            /* Determine range of source neurons for divergent kernel */ \
            int start_s_row = (d_row - row_offset - row_field_size + row_stride) / row_stride; \
            int start_s_col = (d_col - column_offset - column_field_size + column_stride) / column_stride; \
            int end_s_row = (d_row - row_offset) / row_stride; \
            int end_s_col = (d_col - column_offset) / column_stride; \
\
            int weight_offset = to_index * num_weights / to_size; \
\
            /* Iterate over relevant source neurons... */ \
            int k_index = 0; \
            for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) { \
                for (int s_col = start_s_col ; s_col <= end_s_col ; (++s_col, ++k_index)) { \
                    int k_s_row = s_row; \
                    int k_s_col = s_col; \
\
                    /* If wrapping, adjust out of bounds indices accordingly */ \
                    if (wrap) { \
                        k_s_row = (k_s_row < 0) \
                            ? k_s_row + from_rows \
                            : (k_s_row >= from_rows) \
                                ? k_s_row - from_rows : k_s_row; \
    \
                        k_s_col = (k_s_col < 0) \
                            ? k_s_col + from_columns \
                            : (k_s_col >= from_columns) \
                                ? k_s_col - from_columns : k_s_col; \
                    /* Avoid making connections with non-existent neurons */ \
                    } else if (k_s_row < 0 or k_s_row >= from_rows \
                        or k_s_col < 0 or k_s_col >= from_columns) { \
                        continue; \
                    } \
\
                    int from_index = (k_s_row * from_columns) + k_s_col; \
                    int weight_index = weight_offset + k_index; \
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
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const bool wrap = synapse_data.arborized_config.wrap; \
    EXTRACTIONS; \
\
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < to_size) { \
        int d_row = to_index / to_columns; \
        int d_col = to_index % to_columns; \
        NEURON_PRE; \
 \
        /* Determine range of source neurons for divergent kernel */ \
        int start_s_row = (d_row - row_offset - row_field_size + row_stride) / row_stride; \
        int start_s_col = (d_col - column_offset - column_field_size + column_stride) / column_stride; \
        int end_s_row = (d_row - row_offset) / row_stride; \
        int end_s_col = (d_col - column_offset) / column_stride; \
\
        /* Kernels are organized into columns
           One kernel per source neuron */ \
        int kernel_size = row_field_size * column_field_size; \
\
        /* Iterate over relevant source neurons... */ \
        int k_index = 0; \
        for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) { \
            for (int s_col = start_s_col ; s_col <= end_s_col ; (++s_col, ++k_index)) { \
                int k_s_row = s_row; \
                int k_s_col = s_col; \
\
                /* If wrapping, adjust out of bounds indices accordingly */ \
                if (wrap) { \
                    k_s_row = (k_s_row < 0) \
                        ? k_s_row + from_rows \
                        : (k_s_row >= from_rows) \
                            ? k_s_row - from_rows : k_s_row; \
\
                    k_s_col = (k_s_col < 0) \
                        ? k_s_col + from_columns \
                        : (k_s_col >= from_columns) \
                            ? k_s_col - from_columns : k_s_col; \
                /* Avoid making connections with non-existent neurons */ \
                } else if (k_s_row < 0 or k_s_row >= from_rows \
                    or k_s_col < 0 or k_s_col >= from_columns) { \
                    continue; \
                } \
\
                int from_index = (k_s_row * from_columns) + k_s_col; \
\
                /* Row of matrix is the kernel index * row size (see above)
                   Column of matrix is the index of the source neuron */ \
                int weight_index = to_index + (k_index * to_size); \
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

#define CALC_SUBSET(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
SUBSET_PARALLEL(FUNC_NAME##_PARALLEL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
SUBSET_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() {\
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#define CALC_ONE_TO_ONE(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
ONE_TO_ONE_PARALLEL(FUNC_NAME##_PARALLEL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
ONE_TO_ONE_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
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

#define CALC_SUBSET(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
SUBSET_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() { \
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL); \
}

#define CALC_ONE_TO_ONE(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
ONE_TO_ONE_SERIAL(FUNC_NAME##_SERIAL, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
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


#define CALC_ALL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
CALC_FULLY_CONNECTED(FUNC_NAME##_fully_connected, \
    EXTRACTIONS, \
    NEURON_PRE, \
    WEIGHT_OP, \
    NEURON_POST \
); \
CALC_SUBSET(FUNC_NAME##_subset, \
    EXTRACTIONS, \
    NEURON_PRE, \
    WEIGHT_OP, \
    NEURON_POST \
); \
CALC_ONE_TO_ONE(FUNC_NAME##_one_to_one, \
    EXTRACTIONS, \
    NEURON_PRE, \
    WEIGHT_OP, \
    NEURON_POST \
); \
CALC_CONVERGENT(FUNC_NAME##_convergent, \
    EXTRACTIONS, \
    NEURON_PRE, \
    WEIGHT_OP, \
    NEURON_POST \
); \
CALC_DIVERGENT(FUNC_NAME##_divergent, \
    EXTRACTIONS, \
    NEURON_PRE, \
    WEIGHT_OP, \
    NEURON_POST \
);


/******************************************************************************/
/***************** FIRST ORDER CONNECTION ACTIVATOR KERNELS *******************/
/******************************************************************************/

#define CALC_VAL \
    float val = extractor(outputs[from_index], delay) * weights[weight_index];

#define AGGREGATE \
    inputs[to_index] = calc(opcode, inputs[to_index], sum);

#define ACTIVATE_ALL(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_ALL( \
    FUNC_NAME, \
\
    /* EXTRACTIONS */ \
    UPDATE_EXT, \
\
    /* NEURON_PRE
     * Initialize sum to 0.0 */ \
    float sum = 0.0;, \
\
    /* WEIGHT_OP
     * Calculate weight input, add to sum */ \
    CALC_VAL \
    sum += val; \
    UPDATE_CALC, \
\
    /* NEURON_POST
     * Aggregate sum to input */ \
    AGGREGATE) \

/******************************************************************************/
/*************** SECOND ORDER CONNECTION ACTIVATOR KERNELS ********************/
/******************************************************************************/

#define EXTRACT_SECOND_ORDER \
    float * const second_order_weights = synapse_data.second_order_weights.get(); \

#define CALC_VAL_SECOND_ORDER \
    float val = extractor(outputs[from_index], delay) * weights[weight_index]; \
    second_order_weights[weight_index] = \
        calc(opcode, second_order_weights[weight_index], val);

#define ACTIVATE_ALL_SECOND_ORDER(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_ALL(FUNC_NAME, \
    /* EXTRACTIONS */ \
    EXTRACT_SECOND_ORDER; \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE */ \
    , \
 \
    /* WEIGHT_OP
     * Calculate weight input, aggregate to second order buffer */ \
    CALC_VAL_SECOND_ORDER; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST */ \
) \
CALC_ONE_TO_ONE(FUNC_NAME##_convolutional, \
    /* EXTRACTIONS */ \
    EXTRACT_SECOND_ORDER; \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE */ \
    , \
 \
    /* WEIGHT_OP
     * Calculate weight input, aggregate to second order buffer */ \
    CALC_VAL_SECOND_ORDER; \
    UPDATE_CALC;, \
 \
    /* NEURON_POST */ \
); \

#endif
