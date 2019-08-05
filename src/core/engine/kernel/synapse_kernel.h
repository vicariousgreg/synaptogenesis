#ifndef synapse_kernel_h
#define synapse_kernel_h

#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/aggregator.h"
#include "util/parallel.h"
#include "util/constants.h"
#include "util/resources/pointer.h"
#include <omp.h>

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
    const int from_size = synapse_data.from_layer.size; \
    const int from_rows = synapse_data.from_layer.rows; \
    const int from_columns = synapse_data.from_layer.columns; \
    const int to_size = synapse_data.to_layer.size; \
    const int to_rows = synapse_data.to_layer.rows; \
    const int to_columns = synapse_data.to_layer.columns; \
\
    const Opcode opcode = synapse_data.connection.opcode; \
    const int delay = synapse_data.connection.delay; \
    const bool plastic = synapse_data.connection.plastic; \
    const float max_weight = synapse_data.connection.max_weight; \
\
    float * const weights = synapse_data.weights.get(); \
    const int num_weights = synapse_data.num_weights; \
    const int num_weights_per_neuron = num_weights / to_size; \
\
    Output * const outputs = synapse_data.outputs.get(); \
    Output * const destination_outputs = synapse_data.destination_outputs.get(); \
    float * const inputs = synapse_data.inputs.get(); \
\
    const EXTRACTOR extract = synapse_data.extractor; \
    const AGGREGATOR aggregate = synapse_data.aggregator;


// Assembles and defines a serial kernel
#define DEF_KERNEL_SERIAL(PREFIX, FUNC_NAME, EXTRACTIONS, SERIAL_BODY, PARALLEL_BODY) \
HOST void FUNC_NAME##_SERIAL(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    PREFIX##_PREAMBLE; \
    EXTRACTIONS; \
\
    PREFIX##_SERIAL_LOOP_OPEN \
            SERIAL_BODY; \
        } \
    } \
}

// Assembles and defines a parallel kernel
#ifdef __CUDACC__
#define DEF_KERNEL_PARALLEL(PREFIX, FUNC_NAME, EXTRACTIONS, SERIAL_BODY, PARALLEL_BODY) \
GLOBAL void FUNC_NAME##_PARALLEL(SynapseData synapse_data) { \
    SYNAPSE_PREAMBLE; \
    PREFIX##_PREAMBLE; \
    EXTRACTIONS; \
\
    PREFIX##_PARALLEL_LOOP_OPEN \
        PARALLEL_BODY; \
    } \
}

// Parallel stub for non-parallel builds
#else
#define DEF_KERNEL_PARALLEL(PREFIX, FUNC_NAME, EXTRACTIONS, SERIAL_BODY, PARALLEL_BODY) \
static void(*FUNC_NAME##_PARALLEL)(SynapseData) = nullptr;
#endif

// Assembles and defines serial and parallel kernels
#define DEF_KERNELS(PREFIX, FUNC_NAME, EXTRACTIONS, SERIAL_BODY, PARALLEL_BODY) \
DEF_KERNEL_SERIAL(PREFIX, FUNC_NAME, EXTRACTIONS, SERIAL_BODY, PARALLEL_BODY) \
DEF_KERNEL_PARALLEL(PREFIX, FUNC_NAME, EXTRACTIONS, SERIAL_BODY, PARALLEL_BODY)



/*** FULLY CONNECTED **********************************************************/

#define FC_PREAMBLE

#define FC_WEIGHT_LOOP(WEIGHT_INIT, WEIGHT_OP, WEIGHT_INCR) \
{ \
    int weight_index = WEIGHT_INIT; \
    int from_index = 0; \
    for (int from_row = 0 ; from_row < from_rows ; ++from_row) { \
        for (int from_column = 0 ; from_column < from_columns ; ++from_column) { \
            WEIGHT_OP; \
            WEIGHT_INCR; \
            ++from_index; \
        } \
    } \
}

#define FC_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int to_row = 0 ; to_row < to_rows ; ++to_row) { \
    for (int to_column = 0 ; to_column < to_columns ; ++to_column) { \
        int to_index = to_row * to_columns + to_column;

#define FC_PARALLEL_LOOP_OPEN \
int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
if (to_index < to_size) { \
    int to_column = to_index % to_columns; \
    int to_row = to_index / to_columns;

#define DEF_FULLY_CONNECTED(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DEF_KERNELS(FC, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        FC_WEIGHT_LOOP(to_index * from_size, WEIGHT_OP, ++weight_index) \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        FC_WEIGHT_LOOP(to_index, WEIGHT_OP, weight_index += to_size); \
    NEURON_POST; \
)

#define DEF_FULLY_CONNECTED_DUAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP_1, NEURON_MID, WEIGHT_OP_2, NEURON_POST) \
DEF_KERNELS(FC, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        FC_WEIGHT_LOOP(to_index * from_size, WEIGHT_OP_1, ++weight_index) \
    NEURON_MID; \
        FC_WEIGHT_LOOP(to_index * from_size, WEIGHT_OP_2, ++weight_index) \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        FC_WEIGHT_LOOP(to_index, WEIGHT_OP_1, weight_index += to_size); \
    NEURON_MID; \
        FC_WEIGHT_LOOP(to_index, WEIGHT_OP_2, weight_index += to_size); \
    NEURON_POST; \
)

/*** SUBSET *******************************************************************/

#define SUBSET_PREAMBLE \
    const int from_row_start = synapse_data.subset_config.from_row_start; \
    const int from_row_end = synapse_data.subset_config.from_row_end; \
    const int from_col_start = synapse_data.subset_config.from_col_start; \
    const int from_col_end = synapse_data.subset_config.from_col_end; \
    const int to_row_start = synapse_data.subset_config.to_row_start; \
    const int to_row_end = synapse_data.subset_config.to_row_end; \
    const int to_col_start = synapse_data.subset_config.to_col_start; \
    const int to_col_end = synapse_data.subset_config.to_col_end; \
    const int from_kernel_size = synapse_data.subset_config.from_size; \
    const int kernel_to_rows = synapse_data.subset_config.to_row_size; \
    const int kernel_to_cols = synapse_data.subset_config.to_col_size; \
    const int to_kernel_size = synapse_data.subset_config.to_size;

#define SUBSET_WEIGHT_LOOP(WEIGHT_INIT, WEIGHT_OP) \
{ \
    from_kernel_index = 0; \
    for (int from_row = from_row_start ; from_row < from_row_end ; ++from_row) { \
        for (int from_column = from_col_start ; from_column < from_col_end ; ++from_column) { \
            int from_index = from_row * from_columns + from_column; \
            int weight_index = WEIGHT_INIT; \
\
            WEIGHT_OP; \
\
            ++from_kernel_index; \
        } \
    } \
}

#define SUBSET_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int i = 0 ; i < kernel_to_rows ; ++i) { \
    for (int j = 0 ; j < kernel_to_cols ; ++j) { \
        int to_row = i + to_row_start; \
        int to_column = j + to_col_start; \
        int to_kernel_index = i * kernel_to_cols + j; \
        int to_index = to_row * to_columns + to_column; \
        int from_kernel_index = 0;

#define SUBSET_PARALLEL_LOOP_OPEN \
int to_kernel_index = blockIdx.x * blockDim.x + threadIdx.x; \
if (to_kernel_index < to_kernel_size) { \
    int to_row = (to_kernel_index / kernel_to_cols) + to_row_start; \
    int to_column = (to_kernel_index % kernel_to_cols) + to_col_start; \
    int to_index = to_row * to_columns + to_column; \
    int from_kernel_index = 0;

#define DEF_SUBSET(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DEF_KERNELS(SUBSET, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        SUBSET_WEIGHT_LOOP( \
            to_kernel_index * from_kernel_size + from_kernel_index, \
            WEIGHT_OP) \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        SUBSET_WEIGHT_LOOP( \
            from_kernel_index * to_kernel_size + to_kernel_index, \
            WEIGHT_OP) \
    NEURON_POST; \
)

#define DEF_SUBSET_DUAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP_1, NEURON_MID, WEIGHT_OP_2, NEURON_POST) \
DEF_KERNELS(SUBSET, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        SUBSET_WEIGHT_LOOP( \
            to_kernel_index * from_kernel_size + from_kernel_index, \
            WEIGHT_OP_1) \
    NEURON_MID; \
        SUBSET_WEIGHT_LOOP( \
            to_kernel_index * from_kernel_size + from_kernel_index, \
            WEIGHT_OP_2) \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        SUBSET_WEIGHT_LOOP( \
            from_kernel_index * to_kernel_size + to_kernel_index, \
            WEIGHT_OP_1) \
    NEURON_MID; \
        SUBSET_WEIGHT_LOOP( \
            from_kernel_index * to_kernel_size + to_kernel_index, \
            WEIGHT_OP_2) \
    NEURON_POST; \
)

/*** ONE TO ONE ***************************************************************/

#define ONE_PREAMBLE

#define ONE_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int from_row = 0 ; from_row < from_rows ; ++from_row) { \
    for (int from_column = 0 ; from_column < from_columns ; ++from_column) { \
        int to_row = from_row; \
        int to_column = from_column; \
        int from_index = from_row * from_columns + from_column; \
        int to_index = from_index; \
        int weight_index = to_index;

#define ONE_PARALLEL_LOOP_OPEN \
int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
int from_column = to_index % to_columns; \
int from_row = to_index / to_columns; \
if (to_index < from_size) { \
    int to_row = from_row; \
    int to_column = from_column; \
    int from_index = to_index; \
    int weight_index = to_index;

#define DEF_ONE_TO_ONE(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DEF_KERNELS(ONE, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        { WEIGHT_OP; } \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        { WEIGHT_OP; } \
    NEURON_POST; \
)

#define DEF_ONE_TO_ONE_DUAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP_1, NEURON_MID, WEIGHT_OP_2, NEURON_POST) \
DEF_KERNELS(ONE, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        { WEIGHT_OP_1; } \
    NEURON_MID; \
        { WEIGHT_OP_2; } \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        { WEIGHT_OP_1; } \
    NEURON_MID; \
        { WEIGHT_OP_2; } \
    NEURON_POST; \
)

/*** SPARSE *******************************************************************/

#define SPARSE_PREAMBLE \
    int * const nonzero_counts = synapse_data.matrix->nonzero_counts.get(); \
    int * const from_row_indices = synapse_data.matrix->from_row_indices.get(); \
    int * const from_column_indices = synapse_data.matrix->from_column_indices.get(); \
    int * const from_indices = synapse_data.matrix->from_indices.get();

#define SPARSE_WEIGHT_LOOP(WEIGHT_INIT, WEIGHT_OP, WEIGHT_INCR) \
{ \
    int weight_index = WEIGHT_INIT; \
    int nonzero = nonzero_counts[to_index]; \
    for (int i = 0 ; i < nonzero ; ++i) { \
        int from_row = from_row_indices[weight_index]; \
        int from_column = from_column_indices[weight_index]; \
        int from_index = from_indices[weight_index]; \
        WEIGHT_OP; \
        WEIGHT_INCR; \
    } \
}

#define SPARSE_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int to_row = 0 ; to_row < to_rows ; ++to_row) { \
    for (int to_column = 0 ; to_column < to_columns ; ++to_column) { \
        int to_index = to_row * to_columns + to_column;

#define SPARSE_PARALLEL_LOOP_OPEN \
int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
int to_column = to_index % to_columns; \
int to_row = to_index / to_columns; \
if (to_index < to_size) {

#define DEF_SPARSE(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DEF_KERNELS(SPARSE, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        SPARSE_WEIGHT_LOOP( \
            to_index * num_weights_per_neuron, WEIGHT_OP, ++weight_index); \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        SPARSE_WEIGHT_LOOP( \
            to_index, WEIGHT_OP, weight_index += to_size); \
    NEURON_POST; \
)

#define DEF_SPARSE_DUAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP_1, NEURON_MID, WEIGHT_OP_2, NEURON_POST) \
DEF_KERNELS(SPARSE, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        SPARSE_WEIGHT_LOOP( \
            to_index * num_weights_per_neuron, WEIGHT_OP_1, ++weight_index); \
    NEURON_MID; \
        SPARSE_WEIGHT_LOOP( \
            to_index * num_weights_per_neuron, WEIGHT_OP_2, ++weight_index); \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        SPARSE_WEIGHT_LOOP( \
            to_index, WEIGHT_OP_1, weight_index += to_size); \
    NEURON_MID; \
        SPARSE_WEIGHT_LOOP( \
            to_index, WEIGHT_OP_2, weight_index += to_size); \
    NEURON_POST; \
)

/*** CONVERGENT ***************************************************************/

#define CONVERGENT_PREAMBLE \
    const bool convolutional = synapse_data.connection.convolutional; \
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_spacing = synapse_data.arborized_config.row_spacing; \
    const int column_spacing = synapse_data.arborized_config.column_spacing; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const int kernel_size = row_field_size * column_field_size; \
    const bool wrap = synapse_data.arborized_config.wrap; 


#define CONVERGENT_WEIGHT_LOOP(WEIGHT_INIT, WEIGHT_OP) \
{ \
    /* Determine starting row and column for source neurons */ \
    int s_row = to_row * row_stride + (row_spacing * row_offset); \
    int s_col = to_column * column_stride + (column_spacing * column_offset); \
\
    /* Run the kernel */ \
    int k_index = 0; \
    for (int k_row = 0 ; k_row < row_field_size ; ++k_row) { \
        for (int k_col = 0 ; k_col < column_field_size ; (++k_col, ++k_index)) { \
            int from_row = s_row + (k_row * row_spacing); \
            int from_column = s_col + (k_col * column_spacing); \
            int pre_wrap_from_row = from_row; \
            int pre_wrap_from_column = from_column; \
\
            /* If wrapping, adjust out of bounds indices accordingly */ \
            if (wrap) { \
                from_row = \
                    (from_row % from_rows + from_rows) \
                        % from_rows; \
                from_column = \
                    (from_column % from_columns + from_columns) \
                        % from_columns; \
            /* Avoid making connections with non-existent neurons */ \
            } else if (from_row < 0 or from_row >= from_rows \
                or from_column < 0 or from_column >= from_columns) { \
                continue; \
            } \
\
            int from_index = from_row * from_columns + from_column; \
\
            /* Row of matrix is the kernel index * row size (see above) */ \
            WEIGHT_INIT; \
\
            WEIGHT_OP; \
        } \
    } \
}

#define CONVERGENT_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int to_row = 0 ; to_row < to_rows ; ++to_row) { \
    for (int to_column = 0 ; to_column < to_columns ; ++to_column) { \
        int to_index = to_row * to_columns + to_column;

#define CONVERGENT_PARALLEL_LOOP_OPEN \
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < to_size) { \
        int to_row = to_index / to_columns; \
        int to_column = to_index % to_columns;

#define DEF_CONVERGENT(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DEF_KERNELS(CONVERGENT, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        /* Row of matrix is either the first column (convolutional) */ \
        /*   or the index of the destination neuron otherwise */ \
        int weight_offset = (convolutional) ? 0 : (to_index * kernel_size); \
\
        CONVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_offset + k_index;, \
            WEIGHT_OP); \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        /* Column of matrix is either the first column (convolutional) */ \
        /*   or the index of the destination neuron otherwise */ \
        int weight_col = (convolutional) ? 0 : to_index; \
\
        /* Kernels are organized into columns */ \
        /* One kernel per destination neuron */ \
        /*   Unless convolutional (shared kernel) */ \
        int kernel_row_size = (convolutional) ? 1 : to_size; \
\
        CONVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_col + k_index * kernel_row_size;, \
            WEIGHT_OP); \
    NEURON_POST; \
)

#define DEF_CONVERGENT_DUAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP_1, NEURON_MID, WEIGHT_OP_2, NEURON_POST) \
DEF_KERNELS(CONVERGENT, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        int weight_offset = (convolutional) ? 0 : (to_index * kernel_size); \
\
        CONVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_offset + k_index;, \
            WEIGHT_OP_1); \
    NEURON_MID; \
        CONVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_offset + k_index;, \
            WEIGHT_OP_2); \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        int weight_col = (convolutional) ? 0 : to_index; \
        int kernel_row_size = (convolutional) ? 1 : to_size; \
\
        CONVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_col + k_index * kernel_row_size;, \
            WEIGHT_OP_1); \
    NEURON_MID; \
        CONVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_col + k_index * kernel_row_size;, \
            WEIGHT_OP_2); \
    NEURON_POST; \
)

/*** DIVERGENT ****************************************************************/

#define DIVERGENT_PREAMBLE \
    const bool convolutional = synapse_data.connection.convolutional; \
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_spacing = synapse_data.arborized_config.row_spacing; \
    const int column_spacing = synapse_data.arborized_config.column_spacing; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const int kernel_size = row_field_size * column_field_size; \
    const bool wrap = synapse_data.arborized_config.wrap;

#define DIVERGENT_WEIGHT_LOOP(WEIGHT_INIT, WEIGHT_OP) \
{ \
    /* Determine range of source neurons for divergent kernel */ \
    int start_s_row = \
        (to_row + row_spacing * (-row_offset \
            - row_field_size + row_stride)) / row_stride; \
    int start_s_col = \
        (to_column + column_spacing * (-column_offset \
            - column_field_size + column_stride)) / column_stride; \
    int end_s_row = start_s_row + \
        (row_spacing * (row_field_size - row_stride) / row_stride); \
    int end_s_col = start_s_col + \
        (column_spacing * (column_field_size - column_stride) / column_stride); \
\
    /* Iterate over relevant source neurons... */ \
    int k_index = 0; \
    for (int s_row = start_s_row ; s_row <= end_s_row ; (s_row += row_spacing)) { \
        for (int s_col = start_s_col ; s_col <= end_s_col ; (s_col += column_spacing, ++k_index)) { \
            int from_row = s_row; \
            int from_column = s_col; \
            int pre_wrap_from_row = from_row; \
            int pre_wrap_from_column = from_column; \
\
            /* If wrapping, adjust out of bounds indices accordingly */ \
            if (wrap) { \
                from_row = \
                    (from_row % from_rows + from_rows) \
                        % from_rows; \
                from_column = \
                    (from_column % from_columns + from_columns) \
                        % from_columns; \
            /* Avoid making connections with non-existent neurons */ \
            } else if (from_row < 0 or from_row >= from_rows \
                or from_column < 0 or from_column >= from_columns) { \
                continue; \
            } \
\
            int from_index = (from_row * from_columns) + from_column; \
\
            WEIGHT_INIT;\
\
            WEIGHT_OP; \
        } \
    } \
}

#define DIVERGENT_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int to_row = 0 ; to_row < to_rows ; ++to_row) { \
    for (int to_column = 0 ; to_column < to_columns ; ++to_column) { \
        int to_index = to_row * to_columns + to_column;

#define DIVERGENT_PARALLEL_LOOP_OPEN \
int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
if (to_index < to_size) { \
    int to_row = to_index / to_columns; \
    int to_column = to_index % to_columns;

#define DEF_DIVERGENT(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP, NEURON_POST) \
DEF_KERNELS(DIVERGENT, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        int weight_offset = (convolutional) ? 0 : (to_index * (num_weights / to_size)); \
\
        DIVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_offset + k_index;, \
            WEIGHT_OP); \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        DIVERGENT_WEIGHT_LOOP( \
            /* Row of matrix is the kernel index * row size (see above)
               Column of matrix is the index of the source neuron */ \
            int weight_index = (convolutional) ? k_index : (to_index + (k_index * to_size));, \
            WEIGHT_OP); \
    NEURON_POST; \
)

#define DEF_DIVERGENT_DUAL(FUNC_NAME, EXTRACTIONS, NEURON_PRE, WEIGHT_OP_1, NEURON_MID, WEIGHT_OP_2, NEURON_POST) \
DEF_KERNELS(DIVERGENT, FUNC_NAME, EXTRACTIONS, \
    NEURON_PRE; \
        int weight_offset = (convolutional) ? 0 : (to_index * (num_weights / to_size)); \
\
        DIVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_offset + k_index;, \
            WEIGHT_OP_1); \
    NEURON_MID; \
        DIVERGENT_WEIGHT_LOOP( \
            int weight_index = weight_offset + k_index;, \
            WEIGHT_OP_2); \
    NEURON_POST; \
    , \
    NEURON_PRE; \
        DIVERGENT_WEIGHT_LOOP( \
            int weight_index = (convolutional) ? k_index : (to_index + (k_index * to_size));, \
            WEIGHT_OP_1); \
    NEURON_MID; \
        DIVERGENT_WEIGHT_LOOP( \
            int weight_index = (convolutional) ? k_index : (to_index + (k_index * to_size));, \
            WEIGHT_OP_2); \
    NEURON_POST; \
)

/*** CONVERGENT (BY WEIGHT) ***************************************************/

#define CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_PREAMBLE \
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_spacing = synapse_data.arborized_config.row_spacing; \
    const int column_spacing = synapse_data.arborized_config.column_spacing; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const int kernel_size = row_field_size * column_field_size; \
    const bool wrap = synapse_data.arborized_config.wrap;

#define CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP) \
{ \
    for (int to_row = 0 ; to_row < to_rows ; ++to_row) { \
        for (int to_column = 0 ; to_column < to_columns ; ++to_column) { \
            int to_index = to_row * to_columns + to_column; \
\
            /* Determine starting row and column for source neurons */ \
            int s_row = to_row * row_stride + (row_spacing * row_offset); \
            int s_col = to_column * column_stride + (column_spacing * column_offset); \
\
            int from_row = s_row + (k_row * row_spacing); \
            int from_column = s_col + (k_col * column_spacing); \
            int pre_wrap_from_row = from_row; \
            int pre_wrap_from_column = from_column; \
\
            /* If wrapping, adjust out of bounds indices accordingly */ \
            if (wrap) { \
                from_row = \
                    (from_row % from_rows + from_rows) \
                        % from_rows; \
                from_column = \
                    (from_column % from_columns + from_columns) \
                        % from_columns; \
            /* Avoid making connections with non-existent neurons */ \
            } else if (from_row < 0 or from_row >= from_rows \
                or from_column < 0 or from_column >= from_columns) { \
                continue; \
            } \
\
            int from_index = from_row * from_columns + from_column; \
\
            NEURON_OP; \
        } \
    } \
}

#define CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int k_row = 0 ; k_row < row_field_size ; ++k_row) { \
    for (int k_col = 0 ; k_col < column_field_size ; ++k_col) { \
        int weight_index = k_row * column_field_size + k_col;

#define CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_PARALLEL_LOOP_OPEN \
int weight_index = blockIdx.x * blockDim.x + threadIdx.x; \
if (weight_index < (row_field_size * column_field_size)) { \
    int k_row = weight_index / column_field_size; \
    int k_col = weight_index % column_field_size;

#define DEF_CONVERGENT_CONVOLUTIONAL_BY_WEIGHT(FUNC_NAME, EXTRACTIONS, WEIGHT_PRE, NEURON_OP, WEIGHT_POST) \
DEF_KERNELS(CONVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTRACTIONS, \
    WEIGHT_PRE; \
        CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP); \
    WEIGHT_POST; \
    , \
    WEIGHT_PRE; \
        CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP); \
    WEIGHT_POST; \
)

#define DEF_CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_DUAL(FUNC_NAME, EXTRACTIONS, WEIGHT_PRE, NEURON_OP_1, WEIGHT_MID, NEURON_OP_2, WEIGHT_POST) \
DEF_KERNELS(CONVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTRACTIONS, \
    WEIGHT_PRE; \
        CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_1); \
    WEIGHT_MID; \
        CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_2); \
    WEIGHT_POST; \
    , \
    WEIGHT_PRE; \
        CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_1); \
    WEIGHT_MID; \
        CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_2); \
    WEIGHT_POST; \
)

/*** DIVERGENT (BY WEIGHT) ****************************************************/

#define DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_PREAMBLE \
    const int row_field_size = synapse_data.arborized_config.row_field_size; \
    const int column_field_size = synapse_data.arborized_config.column_field_size; \
    const int row_stride = synapse_data.arborized_config.row_stride; \
    const int column_stride = synapse_data.arborized_config.column_stride; \
    const int row_spacing = synapse_data.arborized_config.row_spacing; \
    const int column_spacing = synapse_data.arborized_config.column_spacing; \
    const int row_offset = synapse_data.arborized_config.row_offset; \
    const int column_offset = synapse_data.arborized_config.column_offset; \
    const int kernel_size = row_field_size * column_field_size; \
    const bool wrap = synapse_data.arborized_config.wrap; \

#define DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP) \
{ \
    for (int to_row = 0 ; to_row < to_rows ; ++to_row) { \
        for (int to_column = 0 ; to_column < to_columns ; ++to_column) { \
            int to_index = to_row * to_columns + to_column; \
\
            int s_row = \
                (to_row + row_spacing * (-row_offset \
                    - row_field_size + row_stride)) / row_stride; \
            int s_col = \
                (to_column + column_spacing * (-column_offset \
                    - column_field_size + column_stride)) / column_stride; \
\
            int from_row = s_row + (k_row * row_spacing); \
            int from_column = s_col + (k_col * column_spacing); \
            int pre_wrap_from_row = from_row; \
            int pre_wrap_from_column = from_column; \
\
            /* If wrapping, adjust out of bounds indices accordingly */ \
            if (wrap) { \
                from_row = \
                    (from_row % from_rows + from_rows) \
                        % from_rows; \
                from_column = \
                    (from_column % from_columns + from_columns) \
                        % from_columns; \
            /* Avoid making connections with non-existent neurons */ \
            } else if (from_row < 0 or from_row >= from_rows \
                or from_column < 0 or from_column >= from_columns) { \
                continue; \
            } \
\
            int from_index = (from_row * from_columns) + from_column; \
\
            NEURON_OP; \
        } \
    } \
}

#define DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_SERIAL_LOOP_OPEN \
_Pragma("omp parallel for collapse(2)") \
for (int k_row = 0 ; k_row < row_field_size ; ++k_row) { \
    for (int k_col = 0 ; k_col < column_field_size ; ++k_col) { \
        int weight_index = k_row * column_field_size + k_col;

#define DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_PARALLEL_LOOP_OPEN \
int weight_index = blockIdx.x * blockDim.x + threadIdx.x; \
if (weight_index < (row_field_size * column_field_size)) { \
    int k_row = weight_index / column_field_size; \
    int k_col = weight_index % column_field_size;

#define DEF_DIVERGENT_CONVOLUTIONAL_BY_WEIGHT(FUNC_NAME, EXTRACTIONS, WEIGHT_PRE, NEURON_OP, WEIGHT_POST) \
DEF_KERNELS(DIVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTRACTIONS, \
    WEIGHT_PRE; \
        DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP); \
    WEIGHT_POST; \
    , \
    WEIGHT_PRE; \
        DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP); \
    WEIGHT_POST; \
)

#define DEF_DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_DUAL(FUNC_NAME, EXTRACTIONS, WEIGHT_PRE, NEURON_OP_1, WEIGHT_MID, NEURON_OP_2, WEIGHT_POST) \
DEF_KERNELS(DIVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTRACTIONS, \
    WEIGHT_PRE; \
        DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_1); \
    WEIGHT_MID; \
        DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_2); \
    WEIGHT_POST; \
    , \
    WEIGHT_PRE; \
        DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_1); \
    WEIGHT_MID; \
        DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_NEURON_LOOP(NEURON_OP_2); \
    WEIGHT_POST; \
)



/******************************************************************************/
/******************************************************************************/

/*
 * DEF_CALC take code snippets, create serial and parallel functions, and
 *    wraps them in a getter function that returns a Kernel object.
 * Additional macros are defined for signature consistency with CALC_ALL,
 *    a master macro that calls all of the primary definition macros to create a
 *    collection of functions that implement the code snippets for all primary
 *    connection types.  A function is defined that maps connection types to
 *    getter functions.
 */

#define DEF_CALC(DEF_NAME, FUNC_NAME, EXTR, PRE, OP, POST) \
DEF_##DEF_NAME(FUNC_NAME, EXTR, PRE, OP, POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() {\
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#define DEF_CALC_DUAL(DEF_NAME, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
DEF_##DEF_NAME##_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
static Kernel<SYNAPSE_ARGS> get_##FUNC_NAME() {\
    return Kernel<SYNAPSE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#define CALC_FULLY_CONNECTED(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(FULLY_CONNECTED, FUNC_NAME, EXTR, PRE, OP, POST)
#define CALC_SUBSET(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(SUBSET, FUNC_NAME, EXTR, PRE, OP, POST)
#define CALC_ONE_TO_ONE(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(ONE_TO_ONE, FUNC_NAME, EXTR, PRE, OP, POST)
#define CALC_SPARSE(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(SPARSE, FUNC_NAME, EXTR, PRE, OP, POST)
#define CALC_CONVERGENT(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(CONVERGENT, FUNC_NAME, EXTR, PRE, OP, POST)
#define CALC_DIVERGENT(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(DIVERGENT, FUNC_NAME, EXTR, PRE, OP, POST)
#define CALC_CONVERGENT_CONVOLUTIONAL_BY_WEIGHT(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(CONVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTR, PRE, OP, POST)
#define CALC_DIVERGENT_CONVOLUTIONAL_BY_WEIGHT(FUNC_NAME, EXTR, PRE, OP, POST) \
   DEF_CALC(DIVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTR, PRE, OP, POST)

#define CALC_FULLY_CONNECTED_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(FULLY_CONNECTED, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)
#define CALC_SUBSET_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(SUBSET, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)
#define CALC_ONE_TO_ONE_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(ONE_TO_ONE, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)
#define CALC_SPARSE_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(SPARSE, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)
#define CALC_CONVERGENT_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(CONVERGENT, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)
#define CALC_DIVERGENT_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(DIVERGENT, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)
#define CALC_CONVERGENT_CONVOLUTIONAL_BY_WEIGHT_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(CONVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)
#define CALC_DIVERGENT_CONVOLUTIONAL_BY_WEIGHT_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
   DEF_CALC_DUAL(DIVERGENT_CONVOLUTIONAL_BY_WEIGHT, FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST)

#define CALC_ALL(FUNC_NAME, EXTR, PRE, OP, POST) \
CALC_FULLY_CONNECTED(FUNC_NAME##_fully_connected, EXTR, PRE, OP, POST); \
CALC_SUBSET(         FUNC_NAME##_subset,          EXTR, PRE, OP, POST); \
CALC_ONE_TO_ONE(     FUNC_NAME##_one_to_one,      EXTR, PRE, OP, POST); \
CALC_SPARSE(         FUNC_NAME##_sparse,          EXTR, PRE, OP, POST); \
CALC_CONVERGENT(     FUNC_NAME##_convergent,      EXTR, PRE, OP, POST); \
CALC_DIVERGENT(      FUNC_NAME##_divergent,       EXTR, PRE, OP, POST); \
\
std::map<ConnectionType, Kernel<SYNAPSE_ARGS>> FUNC_NAME##_map = { \
    { FULLY_CONNECTED, get_##FUNC_NAME##_fully_connected() }, \
    { SUBSET,          get_##FUNC_NAME##_subset() }, \
    { ONE_TO_ONE,      get_##FUNC_NAME##_one_to_one() }, \
    { SPARSE,          get_##FUNC_NAME##_sparse() }, \
    { CONVERGENT,      get_##FUNC_NAME##_convergent() }, \
    { DIVERGENT,       get_##FUNC_NAME##_divergent() } \
};

#define CALC_ALL_DUAL(FUNC_NAME, EXTR, PRE, OP1, MID, OP2, POST) \
CALC_FULLY_CONNECTED_DUAL(FUNC_NAME##_fully_connected, EXTR, PRE, OP1, MID, OP2, POST); \
CALC_SUBSET_DUAL(         FUNC_NAME##_subset,          EXTR, PRE, OP1, MID, OP2, POST); \
CALC_ONE_TO_ONE_DUAL(     FUNC_NAME##_one_to_one,      EXTR, PRE, OP1, MID, OP2, POST); \
CALC_SPARSE_DUAL(         FUNC_NAME##_sparse,          EXTR, PRE, OP1, MID, OP2, POST); \
CALC_CONVERGENT_DUAL(     FUNC_NAME##_convergent,      EXTR, PRE, OP1, MID, OP2, POST); \
CALC_DIVERGENT_DUAL(      FUNC_NAME##_divergent,       EXTR, PRE, OP1, MID, OP2, POST); \
\
std::map<ConnectionType, Kernel<SYNAPSE_ARGS>> FUNC_NAME##_map = { \
    { FULLY_CONNECTED, get_##FUNC_NAME##_fully_connected() }, \
    { SUBSET,          get_##FUNC_NAME##_subset() }, \
    { ONE_TO_ONE,      get_##FUNC_NAME##_one_to_one() }, \
    { SPARSE,          get_##FUNC_NAME##_sparse() }, \
    { CONVERGENT,      get_##FUNC_NAME##_convergent() }, \
    { DIVERGENT,       get_##FUNC_NAME##_divergent() } \
};


/******************************************************************************/
/***************** FIRST ORDER CONNECTION ACTIVATOR KERNELS *******************/
/******************************************************************************/

#define CALC_VAL \
    float val = extract(outputs[from_index], delay) * weights[weight_index];

#define AGGREGATE \
    inputs[to_index] = aggregate(inputs[to_index], sum);

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
    float * const second_order_weights = \
        synapse_data.second_order_host_matrix->second_order_weights.get();

#define CALC_VAL_SECOND_ORDER \
    float val = extract(outputs[from_index], delay) * weights[weight_index]; \
    second_order_weights[weight_index] = \
        aggregate(second_order_weights[weight_index], val);

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
CALC_ONE_TO_ONE(FUNC_NAME##_convergent_convolutional, \
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
CALC_ONE_TO_ONE(FUNC_NAME##_divergent_convolutional, \
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

#endif
