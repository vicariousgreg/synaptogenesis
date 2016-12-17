#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "engine/stream_cluster.h"
#include "engine/instruction.h"

class Engine {
    public:
        Engine(Model *model)
                : state(build_state(model)),
                  stream_cluster(model, state) { }

        virtual ~Engine() { delete this->state; }

        void disable_learning();

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_calc_output();
        void stage_send_output();
        void stage_remaining();
        void stage_weights();

        State *state;
        StreamCluster stream_cluster;
};

#endif
