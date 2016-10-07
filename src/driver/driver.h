#ifndef driver_h
#define driver_h

#include "state/state.h"
#include "model/model.h"
#include "parallel.h"
#include "kernel.h"

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
