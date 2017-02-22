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
        StreamCluster(Structure *structure, State *state, Environment *environment)
                : structure(structure),
                  state(state),
                  environment(environment) { }
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
        Structure *structure;
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
        SequentialStreamCluster(Structure *structure, State *state,
            Environment *environment);
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
        FeedforwardStreamCluster(Structure *structure, State *state,
            Environment *environment);

        virtual void launch_weight_update();
};

inline StreamCluster *build_stream_cluster(std::string cluster_name,
        Structure *structure, State *state, Environment *environment) {
    StreamCluster *stream_cluster;

    if (cluster_name == "parallel")
        stream_cluster =
            new ParallelStreamCluster(structure, state, environment);
    else if (cluster_name == "sequential")
        stream_cluster =
            new SequentialStreamCluster(structure, state, environment);
    else if (cluster_name == "feedforward")
        stream_cluster =
            new FeedforwardStreamCluster(structure, state, environment);
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized stream cluster type!");

    return stream_cluster;
}

#endif
