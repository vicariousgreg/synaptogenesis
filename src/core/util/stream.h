#ifndef stream_h
#define stream_h

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

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

        void flush();
        void synchronize();

#ifdef __CUDACC__
        cudaStream_t get_cuda_stream() { return cuda_stream; }
#endif

    protected:
        friend class Event;

        Stream() { }
        DeviceID device_id;
        bool host_flag;

        // Flag for whether worker is running
        bool running;

        // Flag for whether worker is waiting on tasks
        bool waiting;

        // Thread variables
        std::thread thread;
        std::mutex mutex;
        std::queue<std::function<void()>> queue;
        std::condition_variable cv;

        void worker_loop();

#ifdef __CUDACC__
        cudaStream_t cuda_stream;
#endif
};

class DefaultStream : public Stream {
    public:
        DefaultStream(DeviceID device_id, bool host_flag);

        /* Default stream runs everything in one thread */
        virtual void schedule(std::function<void()> f);
        virtual void record(Event *event);
        virtual void wait(Event *event);
};

#endif
