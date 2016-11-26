#include "engine/scheduler.h"
#include "engine/engine.h"

void Scheduler::schedule_execution(Instruction *inst) {
    execute_schedule[inst->connection->to_layer].push_back(inst);
}

void Scheduler::schedule_weight_update(Instruction *inst) {
    weight_update_schedule[inst->connection->to_layer].push_back(inst);
}

void Scheduler::dispatch(Engine *engine) {
    bool done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto it = execute_schedule.begin(); it != execute_schedule.end(); ++it) {
            if (i < it->second.size()) {
                done = false;
                it->second[i]->execute();
            }
        }
    }
    for (auto it = execute_schedule.begin(); it != execute_schedule.end(); ++it) {
        it->second.clear();
    }

    done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto it = weight_update_schedule.begin(); it != weight_update_schedule.end(); ++it) {
            if (i < it->second.size()) {
                done = false;
                it->second[i]->update();
            }
        }
    }
    for (auto it = weight_update_schedule.begin(); it != weight_update_schedule.end(); ++it) {
        it->second.clear();
    }
}
