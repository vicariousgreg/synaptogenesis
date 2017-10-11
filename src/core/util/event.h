#ifndef event_h
#define event_h

#include <thread>
#include <condition_variable>
#include <mutex>

#include "util/parallel.h"

class Stream;
class Scheduler;

class Event {
    public:
        Event(DeviceID device_id, bool host_flag)
                : device_id(device_id), host_flag(host_flag) {
#ifdef __CUDACC__
            if (not host_flag) {
                cudaSetDevice(device_id);
                cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);
            }
#endif
        }

        virtual ~Event() {
#ifdef __CUDACC__
            if (not host_flag) {
                cudaSetDevice(device_id);
                cudaEventDestroy(cuda_event);
            }
#endif
        }

        bool is_host() { return host_flag; }
        DeviceID get_device_id() { return device_id; }

    protected:
        friend class Stream;
        friend class Scheduler;

        DeviceID device_id;
        bool host_flag;

#ifdef __CUDACC__
        cudaEvent_t cuda_event;
#endif
};

#endif
