#ifndef stream_cluster_h
#define stream_cluster_h

#include <map>
#include "model/model.h"
#include "driver/stream.h"
#include "driver/scheduler.h"
#include "util/parallel.h"
#include "util/constants.h"

class Driver;
class StreamCluster;

class StreamCluster {
    public:
        StreamCluster(Model *model, State *state);
        virtual ~StreamCluster();

        void schedule_output_calculations();
        void schedule_non_output_calculations();

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
        void block_state_on_output_calculations();
        void block_state_on_non_output_calculations();
#endif

    private:
        State *state;
        Scheduler *scheduler;
        std::map<Layer*, Stream*> streams[IO_TYPE_SIZE];
};

#endif
