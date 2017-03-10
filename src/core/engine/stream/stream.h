#ifndef stream_h
#define stream_h

#include "engine/stream/event.h"
#include "util/pointer.h"
#include "util/parallel.h"

template <typename... ARGS>
class Kernel;

class Stream {
    public:
        Stream();
        virtual ~Stream();

        template <typename T>
        void transfer(Pointer<T> src, Pointer<T> dst);
        void record(Event *event);
        void wait(Event *event);

        static Stream *get_default_stream();

    protected:
        template <typename... ARGS>
        friend class Kernel;

        static Stream *default_stream;

        Stream(bool is_default_stream);
        bool is_default_stream;
#ifdef __CUDACC__
        cudaStream_t cuda_stream;
#endif
};

template <typename T>
void Stream::transfer(Pointer<T> src, Pointer<T> dst) {
    if (dst.size != src.size)
        ErrorManager::get_instance()->log_error(
            "Attempted to copy memory between pointers of different sizes!");
#ifdef __CUDACC__
    if (src.local and dst.local) memcpy(dst.ptr, src.ptr, src.size * sizeof(T));
    else {
        auto kind = cudaMemcpyDeviceToDevice;

        if (src.local and not dst.local) kind = cudaMemcpyDeviceToHost;
        else if (dst.local) kind = cudaMemcpyHostToDevice;

        if (is_default_stream)
            cudaMemcpyAsync(dst.ptr, src.ptr, src.size * sizeof(T), kind);
        else
            cudaMemcpyAsync(dst.ptr, src.ptr, src.size * sizeof(T), kind, cuda_stream);
    }
#else
    memcpy(dst.ptr, src.ptr, src.size * sizeof(T));
#endif
}

#endif
