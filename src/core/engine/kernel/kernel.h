#ifndef kernel_h
#define kernel_h

#include "util/error_manager.h"
#include "util/stream.h"

template<typename... ARGS>
class Kernel {
    public:
        Kernel() : serial_kernel(nullptr), parallel_kernel(nullptr) { }
        Kernel(void(*serial_kernel)(ARGS...), void (*parallel_kernel)(ARGS...)=nullptr)
                : serial_kernel(serial_kernel),
                  parallel_kernel(parallel_kernel) { }

        void run(Stream *stream, int blocks, int threads, ARGS... args) {
#ifdef __CUDACC__
            if (not stream->is_host()) {
                cudaSetDevice(stream->get_device_id());
                parallel_kernel<<<blocks, threads, 0, stream->get_cuda_stream()>>>(args...);
            } else
#endif
            serial_kernel(args...);
        }

    protected:
        void (*serial_kernel)(ARGS...);
        void (*parallel_kernel)(ARGS...);
};

#endif
