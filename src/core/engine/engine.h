#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "engine/stream_cluster.h"
#include "engine/sequential_stream_cluster.h"
#include "engine/feedforward_stream_cluster.h"
#include "engine/instruction.h"

class Engine {
    public:
        Engine(Model *model, State *state)
                : model(model),
                  state(state),
                  learning_flag(true) { }

        virtual ~Engine() { }

        Buffer *get_buffer() const { return state->get_buffer(); }
        void set_learning_flag(bool status) { learning_flag = status; }

        // Main hooks
        virtual void stage_clear() = 0;
        virtual void stage_input() = 0;
        virtual void stage_output() = 0;
        virtual void stage_calc() = 0;

    protected:
        Model *model;
        State *state;
        bool learning_flag;
};

class ParallelEngine : public Engine {
    public:
        ParallelEngine(Model *model, State *state)
                : Engine(model, state),
                  stream_cluster(new StreamCluster(model, state)) { }
        virtual ~ParallelEngine() { delete this->stream_cluster; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    protected:
        StreamCluster *stream_cluster;
};

class SequentialEngine : public Engine {
    public:
        SequentialEngine(Model *model, State *state)
                : Engine(model, state),
                  stream_cluster(new SequentialStreamCluster(model, state)) { }
        virtual ~SequentialEngine() { delete this->stream_cluster; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    protected:
        SequentialEngine(Model *model, State *state,
            SequentialStreamCluster *stream_cluster)
                : Engine(model, state),
                  stream_cluster(stream_cluster) { }

        SequentialStreamCluster *stream_cluster;
};

class FeedforwardEngine : public SequentialEngine {
    public:
        FeedforwardEngine(Model *model, State *state)
                : SequentialEngine(model, state,
                    new FeedforwardStreamCluster(model, state)) { }
};

#endif
