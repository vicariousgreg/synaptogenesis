/******************************************************************************/
/*************************** CONNECTION KERNELS *******************************/
/******************************************************************************/

/* Parallel and Serial kernels for connection functions.
 * The Parallel versions determine an index based on the CUDA thread/block.
 * The Serial versions perform iterations over all of the neurons.
 *
 * IMPORTANT:
 * Serial implementation is faster if matrix is interpreted in a transposed
 *    fashion compared to parallel.  For serial loops, row is the destination,
 *    column is the source.  This way, inputs to one neuron are contiguous in
 *    memory.
 */


#define FULLY_CONNECTED_SERIAL(function_name, extractions, neuron_op_pre, weight_op, neuron_op_post) \
GLOBAL void function_name(ConnectionData conn_data) { \
    extractions; \
 \
    for (int to_index = 0 ; to_index < conn_data.to_size ; ++to_index) { \
        neuron_op_pre; \
        for (int from_index = 0 ; from_index < conn_data.from_size ; ++from_index) { \
            int weight_index = to_index * conn_data.from_size + from_index; \
            weight_op; \
        } \
        neuron_op_post; \
    } \
}


#define FULLY_CONNECTED_PARALLEL(function_name, extractions, neuron_op_pre, weight_op, neuron_op_post) \
GLOBAL void function_name(ConnectionData conn_data) { \
    extractions; \
 \
    int to_index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (to_index < conn_data.to_size) { \
        neuron_op_pre; \
        for (int from_index = 0 ; from_index < conn_data.from_size ; ++from_index) { \
            int weight_index = from_index * conn_data.to_size + to_index; \
            weight_op; \
        } \
        neuron_op_post; \
    } \
}



#define ONE_TO_ONE_SERIAL(function_name, extractions, weight_op) \
GLOBAL void function_name(ConnectionData conn_data) { \
    extractions; \
 \
    for (int index = 0 ; index < conn_data.to_size ; ++index) { \
        weight_op; \
    } \
}

#define ONE_TO_ONE_PARALLEL(function_name, extractions, weight_op) \
GLOBAL void function_name(ConnectionData conn_data) { \
    extractions; \
 \
    int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index < conn_data.to_size) { \
        weight_op; \
    } \
}



#define CONVERGENT_SERIAL(function_name, extractions, neuron_op_pre, weight_op, neuron_op_post) \
GLOBAL void function_name(ConnectionData conn_data) { \
    extractions; \
 \
    int kernel_size = conn_data.overlap * conn_data.overlap; \
 \
    for (int d_row = 0 ; d_row < conn_data.to_rows ; ++d_row) { \
        for (int d_col = 0 ; d_col < conn_data.to_columns ; ++d_col) { \
            int to_index = d_row * conn_data.to_columns + d_col; \
            neuron_op_pre; \
 \
            /* Determine starting row and column for source neurons */ \
            int s_row = d_row * conn_data.stride - conn_data.fray; \
            int s_col = d_col * conn_data.stride - conn_data.fray; \
 \
            /* Row of matrix is either the first column (convolutional) */ \
            /*   or the index of the destination neuron otherwise */ \
            int weight_offset = (conn_data.convolutional) \
                                ? 0 : to_index * kernel_size; \
 \
            /* Run the kernel */ \
            for (int k_row = 0 ; k_row < conn_data.overlap ; ++k_row) { \
                for (int k_col = 0 ; k_col < conn_data.overlap ; ++k_col) { \
                    int k_s_row = s_row + k_row; \
                    int k_s_col = s_col + k_col; \
 \
                    /* The connection is frayed if the layers are the same size */ \
                    /* Avoid making connections with non-existent neurons! */ \
                    if (conn_data.fray != 0 and (k_s_row < 0 or k_s_row >= conn_data.to_rows \
                        or k_s_col < 0 or k_s_col >= conn_data.to_columns)) \
                        continue; \
 \
                    int from_index = k_s_row * conn_data.from_columns + k_s_col; \
 \
                    /* Column of matrix is the kernel index */ \
                    int weight_col = (k_row * conn_data.overlap) + k_col; \
                    int weight_index = weight_offset + weight_col; \
 \
                    weight_op; \
                } \
            } \
            neuron_op_post; \
        } \
    } \
}

#define CONVERGENT_PARALLEL(function_name, extractions, neuron_op_pre, weight_op, neuron_op_post) \
GLOBAL void function_name(ConnectionData conn_data) { \
    extractions; \
 \
    int kernel_size = conn_data.overlap * conn_data.overlap; \
 \
    int d_row = blockIdx.x * blockDim.x + threadIdx.x; \
    int d_col = blockIdx.y * blockDim.y + threadIdx.y; \
    if (d_row < conn_data.to_rows and d_col < conn_data.to_columns) { \
        int to_index = d_row * conn_data.to_columns + d_col; \
        neuron_op_pre; \
\
        /* Determine starting row and column for source neurons */ \
        int s_row = d_row * conn_data.stride - conn_data.fray; \
        int s_col = d_col * conn_data.stride - conn_data.fray; \
\
        /* Column of matrix is either the first column (convolutional) */ \
        /*   or the index of the destination neuron otherwise */ \
        int weight_col = (conn_data.convolutional) ? 0 : to_index; \
        /* Kernels are organized into columns */ \
        /* One kernel per destination neuron */ \
        /*   Unless convolutional (shared kernel) */ \
        int kernel_row_size = (conn_data.convolutional) \
                              ? 1 : conn_data.to_size; \
\
        /* Run the kernel */ \
        for (int k_row = 0 ; k_row < conn_data.overlap ; ++k_row) { \
            for (int k_col = 0 ; k_col < conn_data.overlap ; ++k_col) { \
                int k_s_row = s_row + k_row; \
                int k_s_col = s_col + k_col; \
\
                /* The connection is frayed if the layers are the same size */ \
                /* Avoid making connections with non-existent neurons! */ \
                if (conn_data.fray != 0 and (k_s_row < 0 or k_s_row >= conn_data.to_rows \
                    or k_s_col < 0 or k_s_col >= conn_data.to_columns)) \
                    continue; \
\
                int from_index = k_s_row * conn_data.from_columns + k_s_col; \
\
                /* Row of matrix is the kernel index * row size (see above) */ \
                int weight_offset = \
                    ((k_row*conn_data.overlap) + k_col) \
                    * kernel_row_size; \
                int weight_index = weight_offset + weight_col; \
\
                weight_op; \
            } \
        } \
        neuron_op_post; \
    } \
}

#ifdef PARALLEL
#define FULLY_CONNECTED FULLY_CONNECTED_PARALLEL
#define ONE_TO_ONE ONE_TO_ONE_PARALLEL
#define CONVERGENT CONVERGENT_PARALLEL
#else
#define FULLY_CONNECTED FULLY_CONNECTED_SERIAL
#define ONE_TO_ONE ONE_TO_ONE_SERIAL
#define CONVERGENT CONVERGENT_SERIAL
#endif

