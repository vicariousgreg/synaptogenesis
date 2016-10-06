#ifndef driver_h
#define driver_h

#include "state/state.h"
#include "model/model.h"
#include "parallel.h"

class Driver {
    public:
        /* Cycles the environment */
        void timestep() {
            this->step_input();
            this->step_output();
            this->step_weights();
        }

        void step_input();
        void print_output();

        /* Activates neural connections, calculating connection input */
        virtual void step_connection_fully_connected(Connection *conn) = 0;
        virtual void step_connection_one_to_one(Connection *conn) = 0;
        virtual void step_connection_divergent(Connection *conn, bool convolutional) = 0;
        virtual void step_connection_convergent(Connection *conn, bool convolutional) = 0;

        /* Calculates neuron outputs */
        virtual void step_output() = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void step_weights() = 0;

        State *state;
        Model *model;
};


#ifdef PARALLEL
template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix(float(*func)(OUT, ARGS...), OUT* outputs, float* weights,
        float* inputs, int from_size, int to_size, Opcode opcode, ARGS... args) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < to_size) {
        float sum = 0;
        for (int row = 0 ; row < from_size ; ++row) {
            sum += func(outputs[row], args...) * weights[row * to_size + col];
        }
        inputs[col] = calc(opcode, inputs[col], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_divergent(float(*func)(OUT, ARGS...), OUT* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional, ARGS... args) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*to_columns + d_col;

    if (d_row < to_rows and d_col < to_columns) {
        float sum = 0.0;

        // Determine range of source neurons for divergent kernel
        int start_s_row = d_row / overlap;
        int start_s_col = d_col / overlap ;
        int end_s_row = (d_row + stride) / overlap;
        int end_s_col = (d_col + stride) / overlap ;

        // Kernels are organized into columns
        // One kernel per source neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = overlap * overlap;
        int kernel_row_size = (convolutional) ? 1 : from_rows * from_columns;

        // Iterate over relevant source neurons...
        for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
            for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                int s_index = (s_row * from_columns) + s_col;
                int k_row = (d_row + ((overlap - stride) * s_row) % overlap);
                int k_col = (d_col + ((overlap - stride) * s_col) % overlap);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row * overlap) + k_col) * kernel_row_size;
                // Column of matrix is either the first column (convolutional)
                //   or the index of the source neuron otherwise
                int weight_col = (convolutional) ? 0 : s_index;

                sum += func(outputs[s_index], args...) *
                    weights[weight_offset + weight_col];
            }
        }
        inputs[d_index] = calc(opcode, inputs[d_index], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_convergent(float(*func)(OUT, ARGS...), OUT* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional, ARGS... args) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*to_columns + d_col;

    if (d_row < to_rows and d_col < to_columns) {
        float sum = 0.0;
        int s_row = d_row * stride;
        int s_col = d_col * stride;

        // Kernels are organized into columns
        // One kernel per destination neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = overlap * overlap;
        int kernel_row_size = (convolutional) ? 1 : to_rows * to_columns;

        // Column of matrix is either the first column (convolutional)
        //   or the index of the destination neuron otherwise
        int weight_col = (convolutional) ? 0 : d_index;

        // Run the kernel
        for (int k_row = 0 ; k_row < overlap ; ++k_row) {
            for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                int s_index = ((s_row+k_row) * from_columns) + (s_col+k_col);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row*overlap) + k_col) * kernel_row_size;
                sum += func(outputs[s_index], args...) *
                    weights[weight_offset + weight_col];
            }
        }
        inputs[d_index] = calc(opcode, inputs[d_index], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void activate_vector(float(*func)(OUT, ARGS...), OUT* outputs, float* weights,
        float* inputs, int size, Opcode opcode, ARGS... args) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        inputs[index] = calc(opcode, inputs[index],
            func(outputs[index], args...) * weights[index]);
    }
}
#else
template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix(float(*func)(OUT, ARGS...), OUT* outputs,
        float* weights, float* inputs, int from_size, int to_size, Opcode opcode, ARGS... args) {
    // IMPORTANT:
    // Serial implementation is faster if matrix is interpreted in a transposed
    //    fashion compared to parallel.  In this loop, row is the destination,
    //    column is the source.  In this way, inputs to one neuron are
    //    contiguous in memory.
    for (int row = 0 ; row < to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < from_size ; ++col) {
            sum += func(outputs[col], args...) *
                weights[row*from_size + col];
        }
        inputs[row] = calc(opcode, inputs[row], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_divergent(float(*func)(OUT, ARGS...), OUT* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional, ARGS... args) {
    int kernel_size = overlap * overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
            int d_index = d_row*to_columns + d_col;
            float sum = 0.0;

            // Determine range of source neurons for divergent kernel
            int start_s_row = d_row / overlap;
            int start_s_col = d_col / overlap ;
            int end_s_row = (d_row + stride) / overlap;
            int end_s_col = (d_col + stride) / overlap ;

            // Iterate over relevant source neurons...
            for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                    int s_index = (s_row * from_columns) + s_col;
                    int k_row = (d_row + ((overlap - stride) * s_row) % overlap);
                    int k_col = (d_col + ((overlap - stride) * s_col) % overlap);

                    // Row of matrix is either the first column (convolutional)
                    //   or the index of the source neuron otherwise
                    int weight_offset = (convolutional) ? 0 : s_index * kernel_size;
                    // Column of matrix is the kernel index
                    int k_index = (k_row * overlap) + k_col;

                    sum += func(outputs[s_index], args...) *
                        weights[weight_offset + k_index];
                }
            }
            inputs[d_index] = calc(opcode, inputs[d_index], sum);
        }
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_convergent(float(*func)(OUT, ARGS...), OUT* outputs, float* weights,
        float* inputs, int from_rows, int from_columns, int to_rows, int to_columns,
        Opcode opcode, int overlap, int stride, bool convolutional, ARGS... args) {
    int kernel_size = overlap * overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
            int d_index = d_row*to_columns + d_col;
            float sum = 0.0;

            // Determine starting row and column for source neurons
            int s_row = d_row * stride;
            int s_col = d_col * stride;

            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (convolutional) ? 0 : d_index * kernel_size;

            // Run the kernel
            for (int k_row = 0 ; k_row < overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < overlap ; ++k_col) {
                    int s_index = ((s_row+k_row) * from_columns) + (s_col+k_col);

                    // Column of matrix is the kernel index
                    int weight_col = (k_row*overlap) + k_col;
                    sum += func(outputs[s_index], args...) *
                        weights[weight_offset + weight_col];
                }
            }
            inputs[d_index] = calc(opcode, inputs[d_index], sum);
        }
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void activate_vector(float(*func)(OUT, ARGS...), OUT* outputs,
        float* weights, float* inputs, int size, Opcode opcode, ARGS... args) {
    for (int index = 0 ; index < size ; ++index) {
        inputs[index] = calc(opcode, inputs[index],
            func(outputs[index], args...) * weights[index]);
    }
}
#endif

template <typename OUT, typename... ARGS>
void step_fully_connected(State* state, Connection *conn, float(*calc_input)(OUT, ARGS...), OUT* outputs, ARGS... args) {
#ifdef PARALLEL
    int blocks = calc_blocks(conn->to_layer->size);
    calc_matrix<OUT, ARGS...><<<blocks, THREADS>>>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->device_input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        conn->opcode,
        args...);
    cudaCheckError("Failed to calculate connection activation!");

#else
    calc_matrix<OUT, ARGS...>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->input + conn->to_layer->index,
        conn->from_layer->size,
        conn->to_layer->size,
        conn->opcode,
        args...);
#endif
}

template <typename OUT, typename... ARGS>
void step_one_to_one(State* state, Connection *conn, float(*calc_input)(OUT, ARGS...), OUT* outputs, ARGS... args) {
#ifdef PARALLEL
    int blocks = calc_blocks(conn->to_layer->size);
    activate_vector<OUT, ARGS...><<<blocks, THREADS>>>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->device_input + conn->to_layer->index,
        conn->to_layer->size,
        conn->opcode,
        args...);
    cudaCheckError("Failed to calculate connection activation!");

#else
    activate_vector<OUT, ARGS...>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->input + conn->to_layer->index,
        conn->to_layer->size,
        conn->opcode,
        args...);
#endif
}

template <typename OUT, typename... ARGS>
void step_divergent(State* state, Connection *conn, bool convolutional, float(*calc_input)(OUT, ARGS...), OUT* outputs, ARGS... args) {
#ifdef PARALLEL
    dim3 blocks_per_grid(
        calc_blocks(conn->to_layer->rows, 1),
        calc_blocks(conn->to_layer->columns, 128));
    dim3 threads_per_block(1, 128);
    calc_matrix_divergent<OUT, ARGS...><<<blocks_per_grid, threads_per_block>>>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->device_input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional,
        args...);
    cudaCheckError("Failed to calculate connection activation!");

#else
    calc_matrix_divergent<OUT, ARGS...>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional,
        args...);
#endif
}

template <typename OUT, typename... ARGS>
void step_convergent(State* state, Connection *conn, bool convolutional, float(*calc_input)(OUT, ARGS...), OUT* outputs, ARGS... args) {
#ifdef PARALLEL
    dim3 blocks_per_grid(
        calc_blocks(conn->to_layer->rows, 1),
        calc_blocks(conn->to_layer->columns, 128));
    dim3 threads_per_block(1, 128);

    calc_matrix_convergent<OUT, ARGS...><<<blocks_per_grid, threads_per_block>>>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->device_input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional,
        args...);
    cudaCheckError("Failed to calculate connection activation!");

#else
    calc_matrix_convergent<OUT, ARGS...>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->input + conn->to_layer->index,
        conn->from_layer->rows,
        conn->from_layer->columns,
        conn->to_layer->rows,
        conn->to_layer->columns,
        conn->opcode,
        conn->overlap,
        conn->stride,
        convolutional,
        args...);
#endif
}

/* Instantiates a driver based on the driver_string in the given model */
Driver* build_driver(Model* model);

#endif
