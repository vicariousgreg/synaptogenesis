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
        virtual void launch_input() = 0;
        virtual void launch_post_input_calculations() = 0;
        virtual void launch_output() = 0;
        virtual void launch_state_update() { }
        virtual void launch_weight_update() = 0;

#ifdef PARALLEL
        virtual void wait_for_input() = 0;
        virtual void wait_for_output() = 0;
#endif

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
        virtual void launch_input();
        virtual void launch_post_input_calculations();
        virtual void launch_output();
        virtual void launch_state_update();
        virtual void launch_weight_update();

#ifdef PARALLEL
        virtual void wait_for_input();
        virtual void wait_for_output();
#endif

    protected:
        InstructionList sort_instructions(IOTypeVector types, bool plastic);

        std::vector<Stream*> streams[sizeof(IOTypes)];

        InstructionList pre_input_instructions;
        InstructionList post_input_instructions;
        InstructionList plastic_instructions;
};

class SequentialStreamCluster : public StreamCluster {
    public:
        SequentialStreamCluster(Model *model, State *state);
        virtual ~SequentialStreamCluster();

        virtual void launch_input();
        virtual void launch_post_input_calculations();
        virtual void launch_output();
        virtual void launch_weight_update();

#ifdef PARALLEL
        virtual void wait_for_input();
        virtual void wait_for_output();

        cudaStream_t compute_cuda_stream;
#endif

    protected:
        std::vector<Stream*> streams;
};

class FeedforwardStreamCluster : public SequentialStreamCluster {
    public:
        FeedforwardStreamCluster(Model *model, State *state);

        virtual void launch_weight_update();
};

#endif
