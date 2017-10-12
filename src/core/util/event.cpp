#include "util/parallel.h"
#include "util/scheduler.h"

Event::Event(DeviceID device_id, bool host_flag)
        : device_id(device_id), host_flag(host_flag) {
    Scheduler::get_instance()->add(this);
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);
    }
#endif
}

Event::~Event() {
    Scheduler::get_instance()->remove(this);
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaEventDestroy(cuda_event);
    }
#endif
}
