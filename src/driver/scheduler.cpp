#include "driver/scheduler.h"

Scheduler *Scheduler::scheduler = 0;

#ifdef PARALLEL
void Scheduler::schedule_execution(cudaStream_t *stream, Instruction *inst) {
    execute_schedule[stream].push_back(inst);
}

void Scheduler::schedule_weight_update(cudaStream_t *stream, Instruction *inst) {
    weight_update_schedule[stream].push_back(inst);
}

#else

void Scheduler::schedule_execution(Instruction *inst) {
    execute_schedule.push_back(inst);
}

void Scheduler::schedule_weight_update(Instruction *inst) {
    weight_update_schedule.push_back(inst);
}

#endif

void Scheduler::dispatch(Driver *driver) {
#ifdef PARALLEL
    bool done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto it = execute_schedule.begin(); it != execute_schedule.end(); ++it) {
            if (i < it->second.size()) {
                done = false;
                driver->curr_stream = it->first;
                driver->step_connection(it->second[i], it->first);
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
                driver->curr_stream = it->first;
                driver->update_weights(it->second[i]);
            }
        }
    }
#else
    for (int i = 0; i < execute_schedule.size(); ++i)
        driver->step_connection(execute_schedule[i]);
    execute_schedule.clear();
    for (int i = 0; i < weight_update_schedule.size(); ++i) {
        driver->update_weights(weight_update_schedule[i]);
    }
    weight_update_schedule.clear();
#endif
}
