#ifndef driver_h
#define driver_h

#include "state/state.h"
#include "model/model.h"
#include "driver/stream_cluster.h"
#include "driver/instruction.h"
#include "util/parallel.h"

class Driver {
    public:
        Driver(Model *model)
                : state(new State(model)),
                  stream_cluster(model, state) { }

        virtual ~Driver() { delete this->state; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_calc_output();
        void stage_send_output();
        void stage_remaining();
        void stage_weights();

        /* Cycles neuron states */
        //virtual void update_state(int start_index, int count) = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        void update_weights(Instruction *inst) { }

        State *state;
        StreamCluster stream_cluster;
};

#endif
