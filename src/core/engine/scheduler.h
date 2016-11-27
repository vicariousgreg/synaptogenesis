#ifndef scheduler_h
#define scheduler_h

#include <vector>
#include <map>
#include "engine/instruction.h"
#include "util/parallel.h"

class Scheduler {
    public:
        void schedule(Instruction *inst);
        void dispatch_execute();
        void dispatch_update();

    private:
        std::map<Layer*, std::vector<Instruction*> > schedules;
};

#endif
