#ifndef sequential_stream_cluster_h
#define sequential_stream_cluster_h

#include "model/model.h"
#include "engine/stream.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/constants.h"

class SequentialStreamCluster {
    public:
        SequentialStreamCluster(Model *model, State *state);
        virtual ~SequentialStreamCluster();

        virtual void launch_calculations();
        virtual void launch_weight_update();

    protected:
#ifdef PARALLEL
        void wait_event(cudaEvent_t *event);
        void block_stream_to(IOType to_type, cudaStream_t cuda_stream);
#endif

        State *state;
        std::vector<Stream*> streams;
};

#endif
