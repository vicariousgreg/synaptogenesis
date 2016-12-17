#ifndef state_h
#define state_h

#include "model/model.h"
#include "io/buffer.h"
#include "engine/kernel/attribute_kernel.h"
#include "state/weight_matrices.h"
#include "util/constants.h"

class State {
    public:
        State(Model *model, OutputType output_type, int matrix_depth);
        virtual ~State();

        /* Resets the state, clearing any non-sensory input
         * If parallel, this will reset cuda events */
        void reset();

        /* Primary environment IO functions.
         *   -> Gather sensory input from the environment buffer
         *   -> Send motor output to the environment buffer
         */
        void transfer_input();
        void transfer_output();

        /* State update functions.
         *  -> update states of a given IOType
         *  -> update states for output (motor) neurons
         *  -> update states for non-output neurons
         *  -> update all states
         */
        void update_states(IOType layer_type);
        void update_output_states();
        void update_non_output_states();
        void update_all_states();

        /* Getters for buffer, matrices */
        Buffer* get_buffer() { return this->buffer; }
        float* get_matrix(int connection_id) {
            return this->weight_matrices->get_matrix(connection_id);
        }

        ////////// FROM STATE /////////////////

        /* Getters for IO data */
        float* get_input() { return input; }
        OutputType get_output_type() { return output_type; }
        Output* get_recent_output() { return recent_output; }
        Output* get_output(int word_index = 0) {
            return output + (total_neurons * word_index);
        }

        /* Updates neuron attributes using engine-specific kernel */
        void update_states(int start_index, int count);

        /* Getters for neuron count related information */
        int get_num_neurons() { return total_neurons; }
        int get_num_neurons(IOType type) { return num_neurons[type]; }
        int get_start_index(IOType type) { return start_indices[type]; }

#ifdef PARALLEL
        /* If parallel, callers may want to wait for IO events */
        void wait_for_input();
        void wait_for_output();

        /* Cuda streams for IO and state computations */
        cudaStream_t io_stream;
        cudaStream_t state_stream;

        /* Cuda events for IO and output events */
        cudaEvent_t
            *input_event,
            *clear_event,
            *output_calc_event,
            *output_event;
#endif

    protected:
        Buffer *buffer;
        WeightMatrices *weight_matrices;

        ////////// FROM STATE /////////////////

#ifdef PARALLEL
        // Pointer to device copy of this object
        State *device_pointer;
#endif

        /* Primary attribute update function */
        ATTRIBUTE_KERNEL attribute_kernel;

        /* IO functions */
        void clear_input();

        // Number of neurons
        int total_neurons;
        int num_neurons[IO_TYPE_SIZE];

        // Start indices by type
        int start_indices[IO_TYPE_SIZE];

        // Neuron input
        float* input;

        // Neuron output
        OutputType output_type;
        Output* output;
        Output* recent_output;
};

State *build_state(Model *model);

#endif
