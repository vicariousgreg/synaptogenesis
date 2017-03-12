#ifndef event_h
#define event_h

#include "util/parallel.h"

class Event {
    public:
        Event();
        virtual ~Event();

        void synchronize();

#ifdef __CUDACC__
        cudaEvent_t cuda_event;
#endif
};

#endif
