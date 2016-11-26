#ifndef scheduler_h
#define scheduler_h

#include <vector>
#include <map>
#include "engine/instruction.h"
#include "util/parallel.h"

class Engine;

class Scheduler {
    public:
        void schedule_execution(Instruction *inst);
        void schedule_weight_update(Instruction *inst);
        void dispatch(Engine *engine);

    private:
        std::map<Layer*, std::vector<Instruction*> > execute_schedule;
        std::map<Layer*, std::vector<Instruction*> > weight_update_schedule;
};

#endif
