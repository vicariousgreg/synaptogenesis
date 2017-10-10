#ifndef event_h
#define event_h

#include <thread>
#include <condition_variable>
#include <mutex>

#include "util/parallel.h"

class Stream;

class Event {
    public:
        Event(DeviceID device_id, bool host_flag);
        virtual ~Event();

        bool is_enqueued();
        void mark_enqueued(Stream *stream);
        void mark_done();

        // Call with nullptr to synchronize
        void wait(Stream *stream);

    protected:
        friend class Stream;

        DeviceID device_id;
        bool host_flag;

        std::mutex mutex;
        std::condition_variable cv;
        bool enqueued;

#ifdef __CUDACC__
        cudaEvent_t cuda_event;
#endif
};

#endif
