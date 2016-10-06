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
        virtual void step_connection(Connection *conn) = 0;

        /* Calculates neuron outputs */
        virtual void step_output() = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void step_weights() = 0;

        State *state;
        Model *model;
};

/* Instantiates a driver based on the driver_string in the given model */
Driver* build_driver(Model* model);

#ifdef PARALLEL
template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < conn.to_size) {
        float sum = 0;
        for (int row = 0 ; row < conn.from_size ; ++row) {
            sum += func(outputs[row], args...) * weights[row * conn.to_size + col];
        }
        inputs[col] = calc(conn.opcode, inputs[col], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_divergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*conn.to_columns + d_col;

    if (d_row < conn.to_rows and d_col < conn.to_columns) {
        float sum = 0.0;

        // Determine range of source neurons for divergent kernel
        int start_s_row = d_row / conn.overlap;
        int start_s_col = d_col / conn.overlap ;
        int end_s_row = (d_row + conn.stride) / conn.overlap;
        int end_s_col = (d_col + conn.stride) / conn.overlap ;

        // Kernels are organized into columns
        // One kernel per source neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = conn.overlap * conn.overlap;
        int kernel_row_size = (conn.convolutional) ? 1 : conn.from_rows * conn.from_columns;

        // Iterate over relevant source neurons...
        for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
            for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                int s_index = (s_row * conn.from_columns) + s_col;
                int k_row = (d_row + ((conn.overlap - conn.stride) * s_row) % conn.overlap);
                int k_col = (d_col + ((conn.overlap - conn.stride) * s_col) % conn.overlap);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row * conn.overlap) + k_col) * kernel_row_size;
                // Column of matrix is either the first column (convolutional)
                //   or the index of the source neuron otherwise
                int weight_col = (conn.convolutional) ? 0 : s_index;

                sum += func(outputs[s_index], args...) *
                    weights[weight_offset + weight_col];
            }
        }
        inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_convergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int d_row = blockIdx.x * blockDim.x + threadIdx.x;
    int d_col = blockIdx.y * blockDim.y + threadIdx.y;
    int d_index = d_row*conn.to_columns + d_col;

    if (d_row < conn.to_rows and d_col < conn.to_columns) {
        float sum = 0.0;
        int s_row = d_row * conn.stride;
        int s_col = d_col * conn.stride;

        // Kernels are organized into columns
        // One kernel per destination neuron
        //   Unless convolutional (shared kernel)
        int kernel_size = conn.overlap * conn.overlap;
        int kernel_row_size = (conn.convolutional) ? 1 : conn.to_rows * conn.to_columns;

        // Column of matrix is either the first column (convolutional)
        //   or the index of the destination neuron otherwise
        int weight_col = (conn.convolutional) ? 0 : d_index;

        // Run the kernel
        for (int k_row = 0 ; k_row < conn.overlap ; ++k_row) {
            for (int k_col = 0 ; k_col < conn.overlap ; ++k_col) {
                int s_index = ((s_row+k_row) * conn.from_columns) + (s_col+k_col);

                // Row of matrix is the kernel index * row size (see above)
                int weight_offset = ((k_row*conn.overlap) + k_col) * kernel_row_size;
                sum += func(outputs[s_index], args...) *
                    weights[weight_offset + weight_col];
            }
        }
        inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_vector(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < conn.from_size) {
        inputs[index] = calc(conn.opcode, inputs[index],
            func(outputs[index], args...) * weights[index]);
    }
}
#else
template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    // IMPORTANT:
    // Serial implementation is faster if matrix is interpreted in a transposed
    //    fashion compared to parallel.  In this loop, row is the destination,
    //    column is the source.  In this way, inputs to one neuron are
    //    contiguous in memory.
    for (int row = 0 ; row < conn.to_size ; ++row) {
        float sum = 0.0;
        for (int col = 0 ; col < conn.from_size ; ++col) {
            sum += func(outputs[col], args...) *
                weights[row*conn.from_size + col];
        }
        inputs[row] = calc(conn.opcode, inputs[row], sum);
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_divergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int kernel_size = conn.overlap * conn.overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < conn.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < conn.to_columns ; ++d_col) {
            int d_index = d_row*conn.to_columns + d_col;
            float sum = 0.0;

            // Determine range of source neurons for divergent kernel
            int start_s_row = d_row / conn.overlap;
            int start_s_col = d_col / conn.overlap ;
            int end_s_row = (d_row + conn.stride) / conn.overlap;
            int end_s_col = (d_col + conn.stride) / conn.overlap ;

            // Iterate over relevant source neurons...
            for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                for (int s_col = start_s_col ; s_col <= end_s_col ; ++s_col) {
                    int s_index = (s_row * conn.from_columns) + s_col;
                    int k_row = (d_row + ((conn.overlap - conn.stride) * s_row) % conn.overlap);
                    int k_col = (d_col + ((conn.overlap - conn.stride) * s_col) % conn.overlap);

                    // Row of matrix is either the first column (convolutional)
                    //   or the index of the source neuron otherwise
                    int weight_offset = (conn.convolutional) ? 0 : s_index * kernel_size;
                    // Column of matrix is the kernel index
                    int k_index = (k_row * conn.overlap) + k_col;

                    sum += func(outputs[s_index], args...) *
                        weights[weight_offset + k_index];
                }
            }
            inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
        }
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_matrix_convergent(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    int kernel_size = conn.overlap * conn.overlap;

    // Iterate over destination neurons
    for (int d_row = 0 ; d_row < conn.to_rows ; ++d_row) {
        for (int d_col = 0 ; d_col < conn.to_columns ; ++d_col) {
            int d_index = d_row*conn.to_columns + d_col;
            float sum = 0.0;

            // Determine starting row and column for source neurons
            int s_row = d_row * conn.stride;
            int s_col = d_col * conn.stride;

            // Row of matrix is either the first column (convolutional)
            //   or the index of the destination neuron otherwise
            int weight_offset = (conn.convolutional) ? 0 : d_index * kernel_size;

            // Run the kernel
            for (int k_row = 0 ; k_row < conn.overlap ; ++k_row) {
                for (int k_col = 0 ; k_col < conn.overlap ; ++k_col) {
                    int s_index = ((s_row+k_row) * conn.from_columns) + (s_col+k_col);

                    // Column of matrix is the kernel index
                    int weight_col = (k_row*conn.overlap) + k_col;
                    sum += func(outputs[s_index], args...) *
                        weights[weight_offset + weight_col];
                }
            }
            inputs[d_index] = calc(conn.opcode, inputs[d_index], sum);
        }
    }
}

template <typename OUT, typename... ARGS>
GLOBAL void calc_vector(float(*func)(OUT, ARGS...),
        OUT* outputs, float* weights, float* inputs,
        Connection conn, ARGS... args) {
    for (int index = 0 ; index < conn.from_size ; ++index) {
        inputs[index] = calc(conn.opcode, inputs[index],
            func(outputs[index], args...) * weights[index]);
    }
}
#endif

template <typename OUT, typename... ARGS>
void step(State* state, Connection *conn, float(*calc_input)(OUT, ARGS...), OUT* outputs, ARGS... args) {
    void(*func)(float(*func)(OUT, ARGS...), OUT*, float*,
        float*, Connection, ARGS...);

    switch (conn->type) {
        case (FULLY_CONNECTED):
            func = &calc_matrix<OUT, ARGS...>;
            break;
        case (ONE_TO_ONE):
            func = &calc_vector<OUT, ARGS...>;
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
            func = &calc_matrix_divergent<OUT, ARGS...>;
            break;
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            func = &calc_matrix_convergent<OUT, ARGS...>;
            break;
        default:
            throw "Unimplemented connection type!";
    }

#ifdef PARALLEL
    dim3 blocks_per_grid;
    dim3 threads_per_block;

    switch (conn->type) {
        case (FULLY_CONNECTED):
        case (ONE_TO_ONE):
            blocks_per_grid = dim3(calc_blocks(conn->to_layer->size));
            threads_per_block = dim3(THREADS);
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            blocks_per_grid = dim3(
                calc_blocks(conn->to_layer->rows, 1),
                calc_blocks(conn->to_layer->columns, 128));
            threads_per_block = dim3(1, 128);
            break;
        default:
            throw "Unimplemented connection type!";
    }

    func<<<blocks_per_grid, threads_per_block>>>(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->device_input + conn->to_layer->index,
        *conn,
        args...);
    cudaCheckError("Failed to calculate connection activation!");

#else
    func(
        calc_input,
        outputs + conn->from_layer->index,
        state->get_matrix(conn->id),
        state->input + conn->to_layer->index,
        *conn,
        args...);
#endif
}

#endif
