#ifndef stream_h
#define stream_h

#include <vector>
#include <map>
#include "model/layer.h"
#include "driver/instruction.h"
#include "parallel.h"
#include "constants.h"

class Driver;
class StreamCluster;

class Stream {
    public:
        Stream(Layer *layer) : to_layer(layer), executed(0) {
            for (int i = 0; i < IO_TYPE_SIZE; ++i)
                last_index[i] = 0;
#ifdef PARALLEL
            cudaStreamCreate(&cuda_stream);
#endif
        }

        void reset() {
            this->executed = 0;
#ifdef PARALLEL
            cudaEventCreateWithFlags(&finished_event, cudaEventDisableTiming);
            for (int i = 0; i < IO_TYPE_SIZE; ++i)
                cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
#endif
        }

        void execute(Driver *driver);
        void execute(Driver *driver, int to_execute);
        void execute(Driver *driver, IOType type);
        void update_weights(Driver *driver);

        void add_instruction(Instruction *inst, IOType from_type) {
            this->last_index[from_type] = instructions.size();
            this->instructions.push_back(inst);
        }

        bool is_done();
        bool is_done(IOType type);
        bool is_running();

#ifdef PARALLEL
        void wait_event(cudaEvent_t event);
#endif

    private:
        friend class StreamCluster;

        int executed;
        int last_index[IO_TYPE_SIZE];
        Layer *to_layer;
        std::vector<Instruction *> instructions;
#ifdef PARALLEL
        cudaEvent_t events[IO_TYPE_SIZE];
        cudaEvent_t finished_event;
        cudaStream_t cuda_stream;
#endif
};

class StreamCluster {
    public:
        StreamCluster() {}

        void add_instruction(Layer *layer, Instruction *inst, IOType from_type) {
            std::map<Layer*, Stream*>::iterator it = streams.find(layer);
            Stream *stream;
            if (it != streams.end()) {
                stream = it->second;
            } else {
                stream = new Stream(layer);
            }
            stream->add_instruction(inst, from_type);
            streams[layer] = stream;
        }

        void reset();
        void execute(Driver *driver);
        void execute(Driver *driver, IOType type);
        void update_weights(Driver *driver);
        bool is_done();
        bool is_done(IOType type);
#ifdef PARALLEL
        void wait_event(cudaEvent_t event);
        void block_stream(cudaStream_t stream);
        void block_stream(cudaStream_t stream, IOType type);
#endif

    private:
        std::map<Layer*, Stream*> streams;
};

#endif
