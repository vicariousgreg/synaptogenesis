#include "util/resources/stream.h"
#include "util/resources/event.h"
#include "util/resources/pointer.h"
#include "util/resources/scheduler.h"

Stream::Stream(DeviceID device_id, bool host_flag)
        : device_id(device_id), host_flag(host_flag) {
    Scheduler::get_instance()->add(this);
#ifdef __CUDACC__
    // Device streams use CUDA streams
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaStreamCreate(&this->cuda_stream);
    } else {
        this->cuda_stream = 0;
    }
#endif
}

Stream::~Stream() {
    Scheduler::get_instance()->remove(this);
#ifdef __CUDACC__
    if (not host_flag and cuda_stream != 0) {
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
