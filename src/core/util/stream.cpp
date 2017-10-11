#include "util/stream.h"
#include "util/event.h"
#include "util/pointer.h"
#include "util/scheduler.h"

Stream::Stream(DeviceID device_id, bool host_flag)
        : device_id(device_id), host_flag(host_flag) {
    Scheduler::get_instance()->add(this);
#ifdef __CUDACC__
    // Device streams use CUDA streams
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaStreamCreate(&this->cuda_stream);
    }
#endif
}

Stream::~Stream() {
#ifdef __CUDACC__
    if (cuda_stream != 0) {
        cudaSetDevice(device_id);
        cudaStreamDestroy(this->cuda_stream);
    }
#endif
}

void Stream::schedule(std::function<void()> f) {
    Scheduler::get_instance()->enqueue_compute(this, f);
}

void Stream::record(Event *event) {
    Scheduler::get_instance()->enqueue_record(this, event);
}

void Stream::wait(Event *event) {
    Scheduler::get_instance()->enqueue_wait(this, event);
}
