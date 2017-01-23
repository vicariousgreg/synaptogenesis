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
        StreamCluster(Model *model, State *state);
        virtual ~StreamCluster();

        void disable_learning();

        void launch_clear_output_calculations();
        void launch_input_output_calculations();
        void launch_non_output_calculations();
        void launch_weight_update();

    private:
        void schedule_from(IOType from_type);
        void schedule_to(IOType to_type);
        void schedule_plastic();
        void sort_schedule(InstructionList &destination);

        void dendrite_DFS(DendriticNode &curr, Stream *stream);

#ifdef PARALLEL
        void wait_event(IOType to_type, cudaEvent_t *event);
        void block_stream_to(IOType to_type, cudaStream_t cuda_stream);
        void block_stream_from(IOType from_type, cudaStream_t cuda_stream);
#endif

        State *state;
        std::map<Layer*, Stream*> streams[IO_TYPE_SIZE];
        InstructionList all_instructions;

        InstructionList clear_output_instructions;
        InstructionList input_output_instructions;
        InstructionList non_output_instructions;
        InstructionList plastic_instructions;

        std::map<Layer*, InstructionList > schedules;
};

#endif
