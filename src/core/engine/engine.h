#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "engine/stream_cluster.h"
#include "engine/instruction.h"

class Engine {
    public:
        Engine(Model *model)
                : model(model),
                  state(new State(model)),
                  stream_cluster(model, state),
                  learning_flag(true) { }

        virtual ~Engine() { delete this->state; }

        Buffer *get_buffer() const { return state->get_buffer(); }
        void set_learning_flag(bool status) { learning_flag = status; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    private:
        Model *model;
        State *state;
        StreamCluster stream_cluster;
        bool learning_flag;
};

#endif
