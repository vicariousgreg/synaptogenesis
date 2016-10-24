#ifndef driver_h
#define driver_h

#include "state/state.h"
#include "model/model.h"
#include "driver/stream.h"
#include "state/izhikevich_attributes.h"
#include "state/rate_encoding_attributes.h"
#include "parallel.h"

class Driver {
    public:
        Driver(Model *model, State *state)
                : state(state), stream_cluster(model, state) { }

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

/* Instantiates a driver based on the driver_string in the given model */
inline Driver* build_driver(Model* model) {
    Attributes *attributes;
    if (model->driver_string == "izhikevich")
        attributes = new IzhikevichAttributes(model);
    else if (model->driver_string == "rate_encoding")
        attributes = new RateEncodingAttributes(model);
    else
        throw "Unrecognized driver type!";
    return new Driver(model, new State(model, attributes, 1));
}

#endif
