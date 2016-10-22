#ifndef driver_h
#define driver_h

#define NUM_KERNEL_STREAMS 3

#include <vector>

#include "state/state.h"
#include "model/model.h"
#include "io/buffer.h"
#include "driver/kernel.h"
#include "driver/instruction.h"
#include "parallel.h"

class Driver {
    public:
        Driver();
        virtual ~Driver() {
            delete this->state;
            for (int i = 0; i < this->all_instructions.size(); ++i)
                delete this->all_instructions[i];
        }

        void build_instructions(Model *model, int timesteps_per_output);

        // Main hooks
        void stage_input(Buffer *buffer);
        void stage_calc_output();
        void stage_send_output(Buffer *buffer);
        void stage_remaining();

        void step_connections(std::vector<Instruction* > instructions);
        void step_all_connections();
        void step_connections_i();
        void step_connections_io();
        void step_connections_xo();
        void step_connections_x();
        void step_connections_other();
        void step_all_states();
        void step_states(IOType layer_type);
        void step_weights();

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

        void clear_input(float* input, int offset, int num_neurons);
        template <typename... ARGS>
        void step(Instruction *inst, float(*calc_input)(Output, ARGS...), ARGS... args);

        State *state;
        std::vector<Instruction* >
            instructions_i,
            instructions_io,
            instructions_xo,
            instructions_x,
            instructions_other;
        std::vector<Instruction* > all_instructions;

#ifdef PARALLEL
        cudaStream_t io_stream;
        cudaStream_t kernel_streams[NUM_KERNEL_STREAMS];
        cudaStream_t *curr_stream;

        cudaEvent_t
            input_stream_event,
            io_event,
            xo_event,
            output_calc_event,
            output_stream_event;
#else
#endif
};

/* Instantiates a driver based on the driver_string in the given model */
Driver* build_driver(Model* model);

inline void Driver::clear_input(float* input, int offset, int num_neurons) {
    int count = num_neurons - offset;
    if (count > 0) {
#ifdef PARALLEL
        int threads = calc_threads(count);
        int blocks = calc_blocks(count);
        // Use the current stream, as set by the driver
        clear_data<<<blocks, threads, 0, *this->curr_stream>>>(
            input + offset,
            num_neurons - offset);
        cudaCheckError("Failed to clear inputs!");
#else
        clear_data(input + offset, count);
#endif
    }
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
void Driver::step(Instruction *inst,
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
    int threads = calc_threads(inst->to_size);

    switch (inst->type) {
        case (FULLY_CONNECTED):
        case (ONE_TO_ONE):
            blocks_per_grid = dim3(calc_blocks(inst->to_size));
            threads_per_block = dim3(threads);
            break;
        case (DIVERGENT):
        case (DIVERGENT_CONVOLUTIONAL):
        case (CONVERGENT):
        case (CONVERGENT_CONVOLUTIONAL):
            blocks_per_grid = dim3(
                inst->to_rows,
                calc_blocks(inst->to_columns));
            threads_per_block = dim3(1, threads);
            break;
        default:
            throw "Unimplemented connection type!";
    }

    // Run the parallel kernel
    // Use the current stream, as set by the driver
    kernel<<<blocks_per_grid, threads_per_block, 0, *this->curr_stream>>>(
        *inst, calc_input, args...);
    cudaCheckError("Failed to calculate connection activation!");

#else
    // Run the serial kernel
    kernel(
        *inst, calc_input, args...);
#endif
}

#endif
