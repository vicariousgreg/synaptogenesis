#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "engine/stream_cluster.h"
#include "engine/sequential_stream_cluster.h"
#include "engine/instruction.h"

class Engine {
    public:
        Engine(Model *model)
                : model(model),
                  state(new State(model)),
                  learning_flag(true) { }

        virtual ~Engine() { delete this->state; }

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
        ParallelEngine(Model *model)
                : Engine(model),
                  stream_cluster(model, state) { }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    private:
        StreamCluster stream_cluster;
};

class SequentialEngine : public Engine {
    public:
        SequentialEngine(Model *model)
                : Engine(model),
                  stream_cluster(model, state) { }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    private:
        SequentialStreamCluster stream_cluster;
};

#endif
