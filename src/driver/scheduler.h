#ifndef scheduler_h
#define scheduler_h

#include <vector>
#include <map>
#include "driver/driver.h"
#include "driver/instruction.h"
#include "parallel.h"

class Scheduler {
    public:
        static Scheduler *get_instance() {
            if (scheduler == NULL)
                scheduler = new Scheduler;
            return scheduler;
        }
#ifdef PARALLEL
        void schedule_execution(cudaStream_t *stream, Instruction *inst);
        void schedule_weight_update(cudaStream_t *stream, Instruction *inst);
#else
        void schedule_execution(Instruction *inst);
        void schedule_weight_update(Instruction *inst);
#endif
        void dispatch(Driver *driver);
    private:
        static Scheduler *scheduler;
        Scheduler() {}

#ifdef PARALLEL
        std::map<cudaStream_t*, std::vector<Instruction*> > execute_schedule;
        std::map<cudaStream_t*, std::vector<Instruction*> > weight_update_schedule;
#else
        std::vector<Instruction*> execute_schedule;
        std::vector<Instruction*> weight_update_schedule;
#endif
};

#endif
