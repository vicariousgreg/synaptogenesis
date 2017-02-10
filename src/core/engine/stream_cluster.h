#ifndef stream_cluster_h
#define stream_cluster_h

#include <map>
#include "model/model.h"
#include "engine/stream.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/constants.h"

class StreamCluster {
    public:
        StreamCluster(Model *model, State *state)
            : model(model),
              state(state) { }
        virtual ~StreamCluster() { }

        virtual void launch_pre_input_calculations() { };
        virtual void launch_post_input_calculations() = 0;
        virtual void launch_weight_update() = 0;

    protected:
        Model *model;
        State *state;
};

typedef std::vector<IOType> IOTypeVector;

class ParallelStreamCluster : public StreamCluster {
    public:
        ParallelStreamCluster(Model *model, State *state);
        virtual ~ParallelStreamCluster();

        virtual void launch_pre_input_calculations();
        virtual void launch_post_input_calculations();
        virtual void launch_weight_update();

    protected:
        InstructionList sort_instructions(IOTypeVector types, bool plastic);

#ifdef PARALLEL
        void wait_event(IOType to_type, cudaEvent_t *event);
        void block_stream_to(IOType to_type, cudaStream_t cuda_stream);
#endif
        std::vector<Stream*> streams[sizeof(IOTypes)];

        InstructionList pre_input_instructions;
        InstructionList post_input_instructions;
        InstructionList plastic_instructions;
};

class SequentialStreamCluster : public StreamCluster {
    public:
        SequentialStreamCluster(Model *model, State *state);
        virtual ~SequentialStreamCluster();

        virtual void launch_post_input_calculations();
        virtual void launch_weight_update();

    protected:
#ifdef PARALLEL
        void wait_event(cudaEvent_t *event);
#endif
        std::vector<Stream*> streams;
};

class FeedforwardStreamCluster : public SequentialStreamCluster {
    public:
        FeedforwardStreamCluster(Model *model, State *state);

        virtual void launch_weight_update();
};

#endif
