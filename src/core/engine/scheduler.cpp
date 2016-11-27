#include "engine/scheduler.h"

void Scheduler::schedule(Instruction *inst) {
    schedules[inst->connection->to_layer].push_back(inst);
}

void Scheduler::dispatch_execute() {
    bool done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto& schedule : this->schedules)
            if (i < schedule.second.size()) {
                done = false;
                schedule.second[i]->execute();
            }
    }
    for (auto& schedule : this->schedules)
        schedule.second.clear();
}

void Scheduler::dispatch_update() {
    bool done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto& schedule : this->schedules)
            if (i < schedule.second.size()) {
                done = false;
                schedule.second[i]->update();
            }
    }
    for (auto& schedule : this->schedules)
        schedule.second.clear();
}
