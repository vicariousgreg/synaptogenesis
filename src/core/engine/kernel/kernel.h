#ifndef kernel_h
#define kernel_h

#include "util/error_manager.h"

template<class K>
class Kernel {
    public:
#ifdef __CUDACC__
        Kernel(K serial_kernel, K parallel_kernel)
                : serial_kernel(serial_kernel),
                  parallel_kernel(parallel_kernel) { }
#else
        Kernel(K kernel)
                : serial_kernel(kernel) { }
#endif

        K get_function(bool parallel) const {
            if (parallel)
#ifdef __CUDACC__
                return parallel_kernel;
#else
                ErrorManager::get_instance()->log_error(
                    "Attempted to retrieve parallel kernel on serial build.");
#endif
            else return serial_kernel;
        }

    protected:
        K serial_kernel;

#ifdef __CUDACC__
        K parallel_kernel;
#endif
};

#endif
