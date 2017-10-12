#ifndef event_h
#define event_h

#include "util/parallel.h"

class Stream;

class Event {
    public:
        Event(DeviceID device_id, bool host_flag);
        virtual ~Event();

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
