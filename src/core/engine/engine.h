#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "io/environment.h"
#include "engine/stream/cluster.h"
#include "engine/stream/instruction.h"

class Engine {
    public:
        Engine(State *state, Environment *environment)
                : state(state),
                  environment(environment),
                  learning_flag(true) {
            for (auto& structure : state->model->get_structures())
                clusters.push_back(build_cluster(
                    structure, state, environment));
        }

        virtual ~Engine() {
            for (auto& cluster : clusters)
                delete cluster;
        }

        void set_learning_flag(bool status) { learning_flag = status; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    protected:
        State *state;
        Environment *environment;
        std::vector<Cluster*> clusters;
        bool learning_flag;
};

#endif
