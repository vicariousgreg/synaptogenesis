#ifndef feedforward_stream_cluster_h
#define feedforward_stream_cluster_h

#include "model/model.h"
#include "engine/sequential_stream_cluster.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/constants.h"

class FeedforwardStreamCluster : public SequentialStreamCluster {
    public:
        FeedforwardStreamCluster(Model *model, State *state);

        void launch_weight_update();
};

#endif
