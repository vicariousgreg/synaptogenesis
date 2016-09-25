#ifndef driver_h
#define driver_h

#include "model.h"

class Driver {
    public:
        virtual void build(Model* model) = 0;

        //virtual void* set_input(int layer_id, void* data) = 0;
        //virtual void* get_output(int layer_id) = 0;

        /* Cycles the environment */
        void timestep() {
            this->step_input();
            this->step_state();
            this->step_output();
            this->step_weights();
        }

        /* Activates neural connections, calculating connection input */
        virtual void step_input() = 0;

        /* Updates neuron state */
        virtual void step_state() = 0;

        /* Calculates neuron outputs */
        virtual void step_output() = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void step_weights() = 0;
};

/* Allocates space for weight matrices and returns a double pointer
 *  containing pointers to starting points for each matrix */
float** build_weight_matrices(Model* model, int depth);

#endif
