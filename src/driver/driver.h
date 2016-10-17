#ifndef driver_h
#define driver_h

#include "state/state.h"
#include "model/model.h"
#include "io/buffer.h"
#include "driver/kernel.h"
#include "parallel.h"

class Driver {
    public:
        void step_input(Buffer *buffer);
        void step_connections();
        void step_output(Buffer *buffer);

        /* Returns the number of bytes taken by output */
        virtual int get_output_size() = 0;

        /* Activates neural connections, calculating connection input */
        virtual void step_connection(Connection *conn) = 0;

        /* Cycles neuron states */
        virtual void step_state() = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void step_weights() = 0;

        State *state;
        Model *model;
};

/* Instantiates a driver based on the driver_string in the given model */
Driver* build_driver(Model* model);

/* Steps activation of a connection.
 * This function is templated to allow for different driver implementations.
 *
 * OUT is the type of output that neurons produce.
 * ARGS is a list of argument types for the input interpreter function.
 *
 * The function takes the following arguments
 *   - state, which contains the tables of neuron properties
 *   - connection specification
 *   - pointer to output list to read
 *   - input calculation function and associated arguments
 *
 */
template <typename OUT, typename... ARGS>
void step(State* state, Connection *conn, OUT* outputs,
        float(*calc_input)(OUT, ARGS...), ARGS... args) {
    void(*kernel)(
        Instruction<OUT>,
        float(*)(OUT, ARGS...), ARGS...);

    // Determine which kernel to use based on connection type
    switch (conn->type) {
        case (FULLY_CONNECTED):
            kernel = &calc_fully_connected<OUT, ARGS...>;
            break;
        case (ONE_TO_ONE):
            kernel = &calc_one_to_one<OUT, ARGS...>;
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
            kernel = &calc_divergent<OUT, ARGS...>;
            break;
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            kernel = &calc_convergent<OUT, ARGS...>;
            break;
        default:
            throw "Unimplemented connection type!";
    }

#ifdef PARALLEL
    // Calculate grid and block sizes based on type
    dim3 blocks_per_grid;
    dim3 threads_per_block;

    switch (conn->type) {
        case (FULLY_CONNECTED):
        case (ONE_TO_ONE):
            blocks_per_grid = dim3(calc_blocks(conn->to_size));
            threads_per_block = dim3(THREADS);
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            blocks_per_grid = dim3(
                calc_blocks(conn->to_rows, 1),
                calc_blocks(conn->to_columns, 128));
            threads_per_block = dim3(1, 128);
            break;
        default:
            throw "Unimplemented connection type!";
    }

    // Run the parallel kernel
    kernel<<<blocks_per_grid, threads_per_block>>>(
        Instruction<OUT>(conn, outputs,
            state->input, get_matrix(conn->id)),
        calc_input, args...);
    cudaCheckError("Failed to calculate connection activation!");

#else
    // Run the serial kernel
    kernel(
        Instruction<OUT>(conn, outputs,
            state->input, state->get_matrix(conn->id)),
        calc_input, args...);
#endif
}

#endif
