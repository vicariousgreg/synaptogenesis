#ifndef stream_cluster_h
#define stream_cluster_h

#include <map>
#include <string>

#include "model/structure.h"
#include "engine/stream.h"
#include "engine/instruction.h"
#include "io/environment.h"
#include "util/parallel.h"
#include "util/constants.h"

class StreamCluster {
    public:
        StreamCluster(State *state, Environment *environment)
                : state(state),
                  environment(environment) { }
        virtual ~StreamCluster() { }

        virtual void launch_pre_input_calculations() { };
        virtual void launch_input() = 0;
        virtual void launch_post_input_calculations() = 0;
        virtual void launch_output() = 0;
        virtual void launch_state_update() { }
        virtual void launch_weight_update() = 0;

#ifdef __CUDACC__
        virtual void wait_for_input() = 0;
        virtual void wait_for_output() = 0;
#endif

    protected:
        State *state;
        Environment *environment;
};

typedef std::vector<IOType> IOTypeVector;

class ParallelStreamCluster : public StreamCluster {
    public:
        ParallelStreamCluster(Structure *structure, State *state,
            Environment *environment);
        virtual ~ParallelStreamCluster();

        virtual void launch_pre_input_calculations();
        virtual void launch_input();
        virtual void launch_post_input_calculations();
        virtual void launch_output();
        virtual void launch_state_update();
        virtual void launch_weight_update();

#ifdef __CUDACC__
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
        SequentialStreamCluster(Structure *structure, State *state,
            Environment *environment);
        virtual ~SequentialStreamCluster();

        virtual void launch_input();
        virtual void launch_post_input_calculations();
        virtual void launch_output();
        virtual void launch_weight_update();

#ifdef __CUDACC__
        virtual void wait_for_input();
        virtual void wait_for_output();

        cudaStream_t compute_cuda_stream;
#endif

    protected:
        std::vector<Stream*> streams;
};

class FeedforwardStreamCluster : public SequentialStreamCluster {
    public:
        FeedforwardStreamCluster(Structure *structure, State *state,
            Environment *environment);

        virtual void launch_weight_update();
};

inline StreamCluster *build_stream_cluster(Structure *structure,
        State *state, Environment *environment) {
    if (not state->check_compatibility(structure))
        ErrorManager::get_instance()->log_error(
            "Stream cluster compatibility conflict detected!");

    switch (structure->stream_type) {
        case(PARALLEL):
            return new ParallelStreamCluster(structure, state, environment);
        case(SEQUENTIAL):
            return new SequentialStreamCluster(structure, state, environment);
        case(FEEDFORWARD):
            return new FeedforwardStreamCluster(structure, state, environment);
        default:
            ErrorManager::get_instance()->log_error(
                "Unrecognized stream cluster type!");
    }
    return NULL;
}

#endif
