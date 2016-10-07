#ifndef state_h
#define state_h

#include "model/model.h"

class State {
    public:
        virtual void build(Model* model) = 0;

        float* get_input();
        void* get_output();
        void set_input(int layer_id, float* input);
        void clear_all_input();
        void clear_input(int layer_id);

        float* get_matrix(int connection_id) {
            return this->weight_matrices[connection_id];
        }

        // Network model
        Model* model;

        // Neuron input
        float* input;

        // Neuron output
        void* output;

#ifdef PARALLEL
        // Locations to store device copies of data.
        // When accessed, these values will be copied here from the device.
        float* device_input;
        void* device_output;
#endif

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
