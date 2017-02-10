#ifndef stream_cluster_h
#define stream_cluster_h

#include <map>
#include "model/model.h"
#include "engine/stream.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/constants.h"

typedef std::vector<IOType> IOTypeVector;

class StreamCluster {
    public:
        StreamCluster(Model *model, State *state);
        virtual ~StreamCluster();

        void launch_non_input_calculations();
        void launch_input_calculations();
        void launch_weight_update();

    private:
        InstructionList sort_instructions(IOTypeVector types, bool plastic);

#ifdef PARALLEL
        void wait_event(IOType to_type, cudaEvent_t *event);
        void block_stream_to(IOType to_type, cudaStream_t cuda_stream);
#endif

        State *state;
        std::vector<Stream*> streams[sizeof(IOTypes)];

        InstructionList input_instructions;
        InstructionList non_input_instructions;
        InstructionList plastic_instructions;
};

#endif
