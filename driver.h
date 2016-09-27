#ifndef driver_h
#define driver_h

#include "state.h"
#include "model.h"

class Driver {
    public:
        void build(Model* model) {
            this->state->build(model);
            this->model = model;
        }

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

        State *state;
        Model *model;
};

#endif
