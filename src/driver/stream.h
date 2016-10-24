#ifndef stream_h
#define stream_h

#include <vector>
#include <map>
#include "model/model.h"
#include "driver/scheduler.h"
#include "driver/instruction.h"
#include "parallel.h"
#include "constants.h"

class Driver;
class StreamCluster;

class Stream {
    public:
        Stream(Layer *layer);
        virtual ~Stream();

        void reset();

        void schedule_execution(Scheduler *scheduler);
        void schedule_execution(int to_schedule, Scheduler *scheduler);
        void schedule_execution(IOType type, Scheduler *scheduler);
        void schedule_weight_update(Scheduler *scheduler);

        void add_instruction(Instruction *inst, IOType from_type) {
            this->last_index[from_type] = instructions.size();
            this->instructions.push_back(inst);
        }

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

class StreamCluster {
    public:
        StreamCluster(Model *model, State *state);
        virtual ~StreamCluster();

        void reset();
        void schedule_execution_from(IOType from_type);
        void schedule_execution_to(IOType to_type);
        void schedule_weight_update();
        void dispatch(Driver *driver);
        bool is_done();
        bool is_done(IOType type);
#ifdef PARALLEL
        void wait_event(IOType to_type, cudaEvent_t *event);
        void block_stream_to(IOType to_type, cudaStream_t stream);
        void block_stream_from(IOType from_type, cudaStream_t stream);
#endif

    private:
        std::map<Layer*, Stream*> streams[IO_TYPE_SIZE];
        Scheduler *scheduler;
};

#endif
