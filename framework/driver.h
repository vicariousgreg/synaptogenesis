#ifndef driver_h
#define driver_h

#include "state.h"
#include "model.h"

class Driver {
    public:
        /* Cycles the environment */
        void timestep() {
            this->step_input();
            this->step_output();
            this->step_weights();
        }

        void step_input();
        void print_output();

        /* Activates neural connections, calculating connection input */
        virtual void step_connection_fully_connected(Connection *conn) = 0;
        virtual void step_connection_one_to_one(Connection *conn) = 0;
        virtual void step_connection_divergent(Connection *conn) = 0;
        virtual void step_connection_convergent(Connection *conn, bool convolutional) = 0;

        /* Calculates neuron outputs */
        virtual void step_output() = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void step_weights() = 0;

        State *state;
        Model *model;
};

/* Instantiates a driver based on the driver_string in the given model */
Driver* build_driver(Model* model);

#endif
