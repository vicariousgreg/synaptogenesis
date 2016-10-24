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

#ifdef PARALLEL
        void initialize();
#endif

        void clear_input();
        void update(int start_index, int count);

        void get_input();
        void send_output();

        void step_output_states();
        void step_non_output_states();
        void step_all_states();
        void step_state(IOType layer_type);

        Buffer* get_buffer() { return this->buffer; }

        float* get_matrix(int connection_id) {
            return this->weight_matrices->get_matrix(connection_id);
        }

        Attributes *attributes;
        Buffer *buffer;

    protected:
        friend class Driver;

        // Weight matrices
        WeightMatrices *weight_matrices;

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
