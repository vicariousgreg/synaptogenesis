#ifndef state_h
#define state_h

#include "model/model.h"
#include "io/buffer.h"
#include "state/weight_matrices.h"
#include "state/attributes.h"
#include "constants.h"

class State {
    public:
        State(Model *model, Attributes *attributes, int weight_depth);
        virtual ~State();

        /* Resets the state, clearing any non-sensory input
         * If parallel, this will reset cuda events */
        void reset();

        /* Gathers sensory input from the environment buffer */
        void get_input();

        /* Sends motor output to the environment buffer */
        void send_output();

#ifdef PARALLEL
        /* If parallel, callers may want to wait for input events */
        void wait_for_input();
        void wait_for_output();
#endif

        /* Updates |count| neuron states from |start_index| */
        void update_states(int start_index, int count);

        /* Updates all states */
        void update_all_states();

        /* Updates states of the given |layer_type| */
        void update_states(IOType layer_type);

        /* Updates the states of all output (motor) neurons */
        void update_output_states();

        /* Updates the states of all non-output (motor) neurons */
        void update_non_output_states();

        Buffer* get_buffer() { return this->buffer; }
        Attributes* get_attributes() { return this->attributes; }

        float* get_matrix(int connection_id) {
            return this->weight_matrices->get_matrix(connection_id);
        }

    protected:
        //friend class Driver;
        friend class StreamCluster;

        // Weight matrices
        WeightMatrices *weight_matrices;
        Buffer *buffer;
        Attributes *attributes;

#ifdef PARALLEL
        cudaStream_t io_stream;
        cudaStream_t state_stream;

        cudaEvent_t
            *input_event,
            *clear_event,
            *output_calc_event,
            *output_event;
#endif
};

#endif
