#ifndef scheduler_h
#define scheduler_h

#include <vector>
#include <map>
#include "engine/instruction.h"
#include "util/parallel.h"

class Engine;

class QueueItem {
    public:
        Instruction *inst;

#ifdef PARALLEL
        QueueItem(Instruction *inst) : inst(inst), event(NULL) {}
        QueueItem(cudaEvent_t *event) : inst(NULL), event(event) {}

        cudaEvent_t *event;
#else
        QueueItem(Instruction *inst) : inst(inst) {}
#endif
};

class Scheduler {
    public:
#ifdef PARALLEL
        void schedule_execution(cudaStream_t *stream, Instruction *inst);
        void schedule_weight_update(cudaStream_t *stream, Instruction *inst);
        void schedule_event(cudaStream_t *stream, cudaEvent_t *event);
#else
        void schedule_execution(Instruction *inst);
        void schedule_weight_update(Instruction *inst);
#endif
        void dispatch(Engine *engine);

    private:
#ifdef PARALLEL
        std::map<cudaStream_t*, std::vector<QueueItem> > execute_schedule;
        std::map<cudaStream_t*, std::vector<Instruction*> > weight_update_schedule;
#else
        std::vector<QueueItem> execute_schedule;
        std::vector<Instruction*> weight_update_schedule;
#endif
};

#endif
