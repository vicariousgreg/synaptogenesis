#ifndef stream_h
#define stream_h

#include "model/layer.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/constants.h"

class Stream {
    public:
        Stream(Layer *layer);
        virtual ~Stream();

        void add_instruction(Instruction *inst, IOType from_type);
        void add_instruction(Instruction *inst);
        void finalize();

        void schedule(InstructionList &schedule);
        void schedule(int to_schedule, InstructionList &schedule);
        void schedule(IOType type, InstructionList &schedule);
        void schedule_plastic(InstructionList &schedule);

#ifdef PARALLEL
        void wait_event(cudaEvent_t *event);
        cudaEvent_t* get_event(IOType type) const { return events[type]; }
        cudaEvent_t* get_finished_event() const { return finished_event; }
#endif

        Layer* const to_layer;

    private:
        int scheduled;
        int last_index[sizeof(IOTypes)];
        InstructionList instructions;

#ifdef PARALLEL
        cudaEvent_t *events[sizeof(IOTypes)];
        cudaEvent_t *finished_event;
        cudaStream_t cuda_stream;
#endif
};

#endif
