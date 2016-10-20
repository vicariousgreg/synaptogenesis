#ifndef driver_h
#define driver_h

#include <vector>

#include "state/state.h"
#include "model/model.h"
#include "io/buffer.h"
#include "driver/kernel.h"
#include "driver/instruction.h"
#include "parallel.h"

class Driver {
    public:
        virtual ~Driver() {
            delete this->state;
        }

        void build_instructions(Model *model, int timesteps_per_output);

        // New Hooks
        void environment_input(Buffer *buffer);
        void null_input();
        void output_calculation();
        void environment_output(Buffer *buffer);
        void input_calculation();
        void internal_calculation();
        void step_weights();

        void step_connections();
        void step_connections(IOType layer_type);
        void step_state();
        void step_state(IOType layer_type);

        /* Returns the output type of the driver */
        virtual OutputType get_output_type() = 0;

        /* Returns the number of timesteps contained in one output */
        virtual int get_timesteps_per_output() = 0;

        /* Activates neural connection, calculating connection input */
        virtual void update_connection(Instruction *inst) = 0;

        /* Cycles neuron states */
        virtual void update_state(int start_index, int count) = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void update_weights(Instruction *inst) = 0;

        State *state;
        std::vector<Instruction* > instructions[IO_TYPE_SIZE];
        std::vector<Instruction* > all_instructions;
};

/* Instantiates a driver based on the driver_string in the given model */
Driver* build_driver(Model* model);

inline void clear_input(float* input, int offset, int num_neurons) {
#ifdef PARALLEL
    int threads = 128;
    int blocks = calc_blocks(num_neurons - offset, threads);
    clear_data<<<blocks, threads>>>(
#else
    clear_data(
#endif
        input + offset,
        num_neurons - offset);
}

/* Steps activation of a connection.
 * This function is templated to allow for different driver implementations.
 *
 * ARGS is a list of argument types for the input interpreter function.
 *
 * The function takes the following arguments
 *   - instruction, which contains all necessary computational data
 *   - pointer to output list to read
 *   - input calculation function and associated arguments
 */
template <typename... ARGS>
void step(Instruction *inst,
        float(*calc_input)(Output, ARGS...), ARGS... args) {
    void(*kernel)(
        Instruction,
        float(*)(Output, ARGS...), ARGS...);

    // Determine which kernel to use based on connection type
    switch (inst->type) {
        case (FULLY_CONNECTED):
            kernel = &calc_fully_connected<ARGS...>;
            break;
        case (ONE_TO_ONE):
            kernel = &calc_one_to_one<ARGS...>;
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
            kernel = &calc_divergent<ARGS...>;
            break;
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            kernel = &calc_convergent<ARGS...>;
            break;
        default:
            throw "Unimplemented connection type!";
    }

#ifdef PARALLEL
    // Calculate grid and block sizes based on type
    dim3 blocks_per_grid;
    dim3 threads_per_block;

    switch (inst->type) {
        case (FULLY_CONNECTED):
        case (ONE_TO_ONE):
            blocks_per_grid = dim3(calc_blocks(inst->to_size));
            threads_per_block = dim3(THREADS);
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            blocks_per_grid = dim3(
                calc_blocks(inst->to_rows, 1),
                calc_blocks(inst->to_columns, 128));
            threads_per_block = dim3(1, 128);
            break;
        default:
            throw "Unimplemented connection type!";
    }

    // Run the parallel kernel
    kernel<<<blocks_per_grid, threads_per_block>>>(
        *inst, calc_input, args...);
    //cudaCheckError("Failed to calculate connection activation!");

#else
    // Run the serial kernel
    kernel(
        *inst, calc_input, args...);
#endif
}

#endif
