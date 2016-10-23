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
        void get_input_from(Buffer *buffer, cudaStream_t stream) {
            this->attributes->get_input_from(buffer, stream);
        }

        void send_output_to(Buffer *buffer, cudaStream_t stream) {
            this->attributes->get_input_from(buffer, stream);
        }

#else
        void get_input_from(Buffer *buffer) {
            this->attributes->get_input_from(buffer);
        }

        void send_output_to(Buffer *buffer) {
            this->attributes->send_output_to(buffer);
        }

#endif

        float* get_matrix(int connection_id) {
            return this->weight_matrices->get_matrix(connection_id);
        }

        Attributes *attributes;

    protected:
        // Weight matrices
        WeightMatrices *weight_matrices;
};

#endif
