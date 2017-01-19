#ifndef stream_h
#define stream_h

#include <vector>
#include "model/layer.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/constants.h"

class Stream {
    public:
        Stream(Layer *layer);
        virtual ~Stream();

        void add_instruction(Instruction *inst, IOType from_type);
        void finalize();

        void schedule(InstructionList &schedule);
        void schedule(int to_schedule, InstructionList &schedule);
        void schedule(IOType type, InstructionList &schedule);
        void schedule_plastic(InstructionList &schedule);

#ifdef PARALLEL
        void wait_event(cudaEvent_t *event);
#endif

    private:
        friend class StreamCluster;

        int scheduled;
        int last_index[IO_TYPE_SIZE];
        Layer *to_layer;
        InstructionList instructions;
#ifdef PARALLEL
        cudaEvent_t *events[IO_TYPE_SIZE];
        cudaEvent_t *finished_event;
        cudaStream_t cuda_stream;
#endif
};

#endif
