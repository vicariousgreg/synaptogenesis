#ifndef state_h
#define state_h

#include "model/model.h"
#include "io/buffer.h"
#include "state/weight_matrices.h"
#include "state/attributes.h"
#include "util/constants.h"

class State {
    public:
        State(Model *model);
        virtual ~State();

        /* Resets the state, clearing any non-sensory input
         * If parallel, this will reset cuda events */
        void reset();

        /* Primary environment IO functions.
         *   -> Gather sensory input from the environment buffer
         *   -> Send motor output to the environment buffer
         */
        void get_input();
        void send_output();

        /* State update functions.
         *  -> update |count| neuron states starting at |start_index|
         *  -> update states of a given IOType
         *  -> update states for output (motor) neurons
         *  -> update states for non-output neurons
         */
        void update_states(int start_index, int count);
        void update_states(IOType layer_type);
        void update_output_states();
        void update_non_output_states();
        void update_all_states();

        /* Getters for buffer, attributes, matrices */
        Attributes* get_attributes() { return this->attributes; }
        Buffer* get_buffer() { return this->buffer; }
        float* get_matrix(int connection_id) {
            return this->weight_matrices->get_matrix(connection_id);
        }

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
        Attributes *attributes;
        Buffer *buffer;
        WeightMatrices *weight_matrices;
};

#endif
