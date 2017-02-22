#ifndef engine_h
#define engine_h

#include "state/state.h"
#include "model/model.h"
#include "engine/stream_cluster.h"
#include "engine/instruction.h"

class Engine {
    public:
        Engine(Model *model, State *state)
                : model(model),
                  state(state),
                  learning_flag(true) {
            for (auto& structure : model->get_structures())
                stream_clusters.push_back(state->build_stream_cluster(structure));
        }

        virtual ~Engine() {
            for (auto& cluster : stream_clusters)
                delete cluster;
        }

        Buffer *get_buffer(Structure *structure) const {
            return state->get_buffer(structure);
        }

        void set_learning_flag(bool status) { learning_flag = status; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    protected:
        Model *model;
        State *state;
        std::vector<StreamCluster*> stream_clusters;
        bool learning_flag;
};

#endif
