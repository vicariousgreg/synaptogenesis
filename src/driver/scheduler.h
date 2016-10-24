#ifndef scheduler_h
#define scheduler_h

#include <vector>
#include <map>
#include "driver/instruction.h"
#include "parallel.h"

class Driver;

class Scheduler {
    public:
#ifdef PARALLEL
        void schedule_execution(cudaStream_t *stream, Instruction *inst);
        void schedule_weight_update(cudaStream_t *stream, Instruction *inst);
#else
        void schedule_execution(Instruction *inst);
        void schedule_weight_update(Instruction *inst);
#endif
        void dispatch(Driver *driver);

    private:
#ifdef PARALLEL
        std::map<cudaStream_t*, std::vector<Instruction*> > execute_schedule;
        std::map<cudaStream_t*, std::vector<Instruction*> > weight_update_schedule;
#else
        std::vector<Instruction*> execute_schedule;
        std::vector<Instruction*> weight_update_schedule;
#endif
};

#endif
