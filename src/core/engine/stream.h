#ifndef stream_h
#define stream_h

#include <vector>
#include "model/layer.h"
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

        void schedule(StreamCluster *stream_cluster);
        void schedule(int to_schedule, StreamCluster *stream_cluster);
        void schedule(IOType type, StreamCluster *stream_cluster);
        void schedule_plastic(StreamCluster *stream_cluster);

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
