#ifndef event_h
#define event_h

#include "util/parallel.h"

class Event {
    public:
        Event(DeviceID device_id, bool host_flag);
        virtual ~Event();

        void synchronize();

#ifdef __CUDACC__
        cudaEvent_t get_cuda_event() { return cuda_event; }
#endif

    protected:
        DeviceID device_id;
        bool host_flag;

#ifdef __CUDACC__
        cudaEvent_t cuda_event;
#endif
};

#endif
