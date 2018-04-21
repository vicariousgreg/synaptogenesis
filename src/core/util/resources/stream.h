#ifndef stream_h
#define stream_h

#include <functional>

#include "util/parallel.h"

class Event;

class Stream {
    public:
        Stream(DeviceID device_id, bool host_flag);
        virtual ~Stream();

        virtual void schedule(std::function<void()> f);
        virtual void record(Event *event);
        virtual void wait(Event *event);

        bool is_host() { return host_flag; }
        DeviceID get_device_id() { return device_id; }

#ifdef __CUDACC__
        cudaStream_t get_cuda_stream() { return cuda_stream; }
#endif

    protected:
        friend class Event;
        friend class Scheduler;

        Stream() { }
        DeviceID device_id;
        bool host_flag;

#ifdef __CUDACC__
        cudaStream_t cuda_stream;
#endif
};

class DefaultStream : public Stream {
    public:
        DefaultStream(DeviceID device_id, bool host_flag) {
            this->device_id = device_id;
            this->host_flag = host_flag;
#ifdef __CUDACC__
            this->cuda_stream = 0;
#endif
        }

        /* Default stream runs everything in one thread */
        virtual void schedule(std::function<void()> f) { f(); }
        virtual void record(Event *event) { }
        virtual void wait(Event *event) { }
};

#endif
