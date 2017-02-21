#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "engine/stream_cluster.h"
#include "engine/instruction.h"

class Engine {
    public:
        Engine(Model *model, State *state,
            StreamCluster *stream_cluster)
                : model(model),
                  state(state),
                  stream_cluster(stream_cluster),
                  learning_flag(true) { }

        virtual ~Engine() { delete this->stream_cluster; }

        Buffer *get_buffer() const { return state->get_buffer(); }
        void set_learning_flag(bool status) { learning_flag = status; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    protected:
        Model *model;
        State *state;
        StreamCluster *stream_cluster;
        bool learning_flag;
};

class ParallelEngine : public Engine {
    public:
        ParallelEngine(Model *model, State *state)
            : Engine(model, state,
                new ParallelStreamCluster(model, state)) { }
};

class SequentialEngine : public Engine {
    public:
        SequentialEngine(Model *model, State *state)
            : Engine(model, state,
                new SequentialStreamCluster(model, state)) { }
};

class FeedforwardEngine : public Engine {
    public:
        FeedforwardEngine(Model *model, State *state)
            : Engine(model, state,
                new FeedforwardStreamCluster(model, state)) { }
};

#endif
