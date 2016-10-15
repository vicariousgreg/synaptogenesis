#ifndef state_h
#define state_h

#include "model/model.h"
#include "io/buffer.h"

class State {
    public:
        virtual void build(Model* model, int output_size) = 0;

        void copy_input(Buffer *buffer);
        void copy_output(Buffer *buffer);

        float* get_matrix(int connection_id) {
            return this->weight_matrices[connection_id];
        }

        // Network model
        Model* model;

        // Size of output
        int output_size;

        // Neuron input
        float* input;

        // Neuron output
        void* output;

    protected:
        // Weight matrices
        float** weight_matrices;
};

/* Allocates space for weight matrices and returns a double pointer
 *  containing pointers to starting points for each matrix */
float** build_weight_matrices(Model* model, int depth);

/* Allocates data on the host */
void* allocate_host(int count, int size);

#ifdef PARALLEL
/* Allocates data on the device */
void* allocate_device(int count, int size, void* source);
#endif

#endif
