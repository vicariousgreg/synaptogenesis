#ifndef driver_h
#define driver_h

#include "model.h"

class Driver {
    public:
        virtual void build(Model* model) = 0;

        virtual void set_input(int layer_id, float* data) = 0;
        virtual void randomize_input(int layer_id, float max) = 0;
        virtual void clear_input(int layer_id) = 0;
        virtual float* get_input() = 0;

        //virtual void* get_output(int layer_id) = 0;

        /* Cycles the environment */
        void timestep() {
            this->step_input();
            this->step_output();
            this->step_weights();
        }

        /* Activates neural connections, calculating connection input */
        virtual void step_input() = 0;

        /* Calculates neuron outputs */
        virtual void step_output() = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void step_weights() = 0;

        // Network model
        Model* model;

    protected:
        // Neuron input
        float *input;

        // Weight matrices
        float** weight_matrices;

#ifdef PARALLEL
        // Locations to store device copies of data.
        // When accessed, these values will be copied here from the device.
        float *device_input;
#endif
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
