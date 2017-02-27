#ifndef stream_h
#define stream_h

#include "engine/stream/event.h"
#include "util/parallel.h"

class Stream {
    public:
        Stream();
        virtual ~Stream();

        virtual void record(Event *event);
        virtual void wait(Event *event);

        template <typename... ARGS>
        void run_kernel(void (*kernel)(ARGS...),
            int blocks, int threads, ARGS... args);

        static Stream *get_default_stream();

    protected:
        static Stream *default_stream;

        Stream(bool is_default_stream);
        bool is_default_stream;
#ifdef __CUDACC__
        cudaStream_t cuda_stream;
#endif
};

class DefaultStream : public Stream {
    public:
        DefaultStream() : Stream(true) { }
        virtual void record(Event *event);
        virtual void wait(Event *event);
};

template <typename... ARGS>
void Stream::run_kernel(void (*kernel)(ARGS...),
        int blocks, int threads, ARGS... args) {
#ifdef __CUDACC__
    kernel<<<blocks, threads, 0, cuda_stream>>>(args...);
#else
    kernel(args...);
#endif
}

#endif
