#include "engine/scheduler.h"
#include "engine/engine.h"

#ifdef PARALLEL
void Scheduler::schedule_execution(cudaStream_t *stream, Instruction *inst) {
    execute_schedule[stream].push_back(QueueItem(inst));
}

void Scheduler::schedule_weight_update(cudaStream_t *stream, Instruction *inst) {
    weight_update_schedule[stream].push_back(inst);
}

void Scheduler::schedule_event(cudaStream_t *stream, cudaEvent_t *event) {
    execute_schedule[stream].push_back(QueueItem(event));
}

#else

void Scheduler::schedule_execution(Instruction *inst) {
    execute_schedule.push_back(QueueItem(inst));
}

void Scheduler::schedule_weight_update(Instruction *inst) {
    weight_update_schedule.push_back(inst);
}

#endif

void Scheduler::dispatch(Engine *engine) {
#ifdef PARALLEL
    bool done = false;
    for (int i = 0; not done; ++i) {
        done = true;
        for (auto it = execute_schedule.begin(); it != execute_schedule.end(); ++it) {
            if (i < it->second.size()) {
                cudaStream_t *stream = it->first;
                QueueItem &item = it->second[i];
                done = false;
                if (item.inst != NULL)
                    item.inst->execute(stream);
                else if (item.event != NULL)
                    cudaEventRecord(*item.event, *stream);
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
                it->second[i]->update(it->first);
            }
        }
    }
#else
    for (int i = 0; i < execute_schedule.size(); ++i) {
        execute_schedule[i].inst->execute();
    }
    execute_schedule.clear();
    for (int i = 0; i < weight_update_schedule.size(); ++i) {
        weight_update_schedule[i]->update();
    }
    weight_update_schedule.clear();
#endif
}
