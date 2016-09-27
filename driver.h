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

        void step_input() {
            for (int cid = 0 ; cid < this->model->num_connections; ++cid) {
                Connection &conn = this->model->connections[cid];
                if (conn.type == FULLY_CONNECTED) {
                    step_connection_fully_connected(conn);
                } else if (conn.type == ONE_TO_ONE) {
                    step_connection_one_to_one(conn);
                }
            }
        }

        /* Activates neural connections, calculating connection input */
        virtual void step_connection_fully_connected(Connection &conn) = 0;
        virtual void step_connection_one_to_one(Connection &conn) = 0;

        /* Calculates neuron outputs */
        virtual void step_output() = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        virtual void step_weights() = 0;

        State *state;
        Model *model;
};

#endif
