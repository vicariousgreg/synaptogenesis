#include "util/event.h"

Event::Event(DeviceID device_id, bool host_flag)
        : device_id(device_id), host_flag(host_flag) {
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);
    }
#endif
}

Event::~Event() {
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaEventDestroy(cuda_event);
    }
#endif
}

void Event::synchronize() {
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaEventSynchronize(cuda_event);
    }
#endif
}
