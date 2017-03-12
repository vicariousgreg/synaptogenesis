#ifndef stream_h
#define stream_h

#include "util/event.h"
#include "util/parallel.h"

template <typename... ARGS>
class Kernel;

template <typename T>
class Pointer;

class Stream {
    public:
        Stream();
        virtual ~Stream();

        void record(Event *event);
        void wait(Event *event);

    protected:
        template <typename... ARGS>
        friend class Kernel;

        template <typename T>
        friend class Pointer;

#ifdef __CUDACC__
        cudaStream_t cuda_stream;
#endif
};

class DefaultStream : public Stream {
    public:
        DefaultStream();
};

#endif
