#ifndef kernel_h
#define kernel_h

#include "util/error_manager.h"
#include "engine/stream/stream.h"

template<typename... ARGS>
class Kernel {
    public:
#ifdef __CUDACC__
        Kernel() : serial_kernel(nullptr), parallel_kernel(nullptr) { }
        Kernel(void(*serial_kernel)(ARGS...), void (*parallel_kernel)(ARGS...))
                : serial_kernel(serial_kernel),
                  parallel_kernel(parallel_kernel) { }
#else
        Kernel() : serial_kernel(nullptr) { }
        Kernel(void(*kernel)(ARGS...))
                : serial_kernel(kernel) { }
#endif

        void run(Stream *stream, int blocks, int threads, ARGS... args) {
#ifdef __CUDACC__
            if (stream->is_default_stream)
                parallel_kernel<<<blocks, threads>>>(args...);
            else
                parallel_kernel<<<blocks, threads, 0, stream->cuda_stream>>>(args...);
#else
            serial_kernel(args...);
#endif
        }

    protected:
        void (*serial_kernel)(ARGS...);

#ifdef __CUDACC__
        void (*parallel_kernel)(ARGS...);
#endif
};

#endif
