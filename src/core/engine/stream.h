#ifndef stream_h
#define stream_h

#include <vector>
#include "model/layer.h"
#include "engine/scheduler.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/constants.h"

class Engine;
class StreamCluster;

class Stream {
    public:
        Stream(Layer *layer);
        virtual ~Stream();

        void add_instruction(Instruction *inst, IOType from_type);
        void finalize();
        void reset();

        void schedule_execution(Scheduler *scheduler);
        void schedule_execution(int to_schedule, Scheduler *scheduler);
        void schedule_execution(IOType type, Scheduler *scheduler);
        void schedule_weight_update(Scheduler *scheduler);

        bool is_done();
        bool is_done(IOType type);
        bool is_running();

#ifdef PARALLEL
        void wait_event(cudaEvent_t *event);
#endif

    private:
        friend class StreamCluster;

        int scheduled;
        int last_index[IO_TYPE_SIZE];
        Layer *to_layer;
        std::vector<Instruction *> instructions;
#ifdef PARALLEL
        cudaEvent_t *events[IO_TYPE_SIZE];
        cudaEvent_t *finished_event;
        cudaStream_t cuda_stream;
#endif
};

#endif
