#ifndef state_h
#define state_h

#include "model/model.h"
#include "io/buffer.h"
#include "constants.h"

class State {
    public:
        State(Model *model, int weight_depth);
        virtual ~State();

        void get_input_from(Buffer *buffer);
        void send_output_to(Buffer *buffer);

        float* get_matrix(int connection_id) {
            return this->weight_matrices[connection_id];
        }

        // Number of neurons
        int total_neurons;
        int num_neurons[LAYER_TYPE_SIZE];
        int start_index[LAYER_TYPE_SIZE];

        // Neuron input
        float* input;

        // Neuron output
        Output* output;
        Output* recent_output;

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
