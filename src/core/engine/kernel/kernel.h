#ifndef kernel_h
#define kernel_h

#include "engine/kernel/synapse_data.h"
#include "network/connection.h"
#include "network/dendritic_node.h"
#include "util/stream.h"
#include "util/pointer.h"

template<typename... ARGS>
class Kernel {
    public:
        Kernel() : serial_kernel(nullptr), parallel_kernel(nullptr) { }
        Kernel(void(*serial_kernel)(ARGS...),
               void (*parallel_kernel)(ARGS...)=nullptr)
                : serial_kernel(serial_kernel),
                  parallel_kernel(parallel_kernel) { }

        void run(Stream *stream, int blocks, int threads, ARGS... args) {
#ifdef __CUDACC__
            if (not stream->is_host()) {
                cudaSetDevice(stream->get_device_id());
                if (parallel_kernel == nullptr)
                    ErrorManager::get_instance()->log_error(
                        "Attempted to run nullptr kernel!");
                else
                    parallel_kernel
                    <<<blocks, threads, 0, stream->get_cuda_stream()>>>
                        (args...);
            } else
#endif
                if (serial_kernel == nullptr)
                    ErrorManager::get_instance()->log_error(
                        "Attempted to run nullptr kernel!");
                else
                    serial_kernel(args...);
        }

        bool is_null() { return serial_kernel == nullptr; }

    protected:
        void (*serial_kernel)(ARGS...);
        void (*parallel_kernel)(ARGS...);
};

/* Sets input data (use val=0.0 for clear) */
Kernel<float, Pointer<float>, int> get_set_data();

/* Randomizes input data */
Kernel<Pointer<float>, int, float, float, bool> get_randomize_data_normal();
Kernel<Pointer<float>, int, float, float, bool> get_randomize_data_poisson();

/* Dendritic tree internal computation */
Kernel<int, Pointer<float>, Pointer<float>, bool> get_calc_internal();
Kernel<int, int, Pointer<float>, Pointer<float>>
get_calc_internal_second_order();

/* Base activator kernel */
Kernel<SYNAPSE_ARGS> get_base_activator_kernel(Connection *conn);

#endif
