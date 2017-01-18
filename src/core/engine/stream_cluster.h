#ifndef stream_cluster_h
#define stream_cluster_h

#include <vector>
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

        void reset();
        void disable_learning();

        void schedule_clear_output_calculations();
        void schedule_input_output_calculations();
        void schedule_non_output_calculations();
        void schedule_weight_update();

    private:
        friend class Stream;
        void schedule(Instruction *inst);

        void schedule_from(IOType from_type);
        void schedule_to(IOType to_type);

        void dispatch_activate();
        void dispatch_update();

#ifdef PARALLEL
        void wait_event(IOType to_type, cudaEvent_t *event);
        void block_stream_to(IOType to_type, cudaStream_t cuda_stream);
        void block_stream_from(IOType from_type, cudaStream_t cuda_stream);
#endif

        State *state;
        std::map<Layer*, Stream*> streams[IO_TYPE_SIZE];
        std::vector<Instruction*> all_instructions;

        std::map<Layer*, std::vector<Instruction*> > schedules;
};

#endif
