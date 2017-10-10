#include "util/event.h"
#include "util/stream.h"

Event::Event(DeviceID device_id, bool host_flag)
        : device_id(device_id), host_flag(host_flag), enqueued(false) {
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

bool Event::is_enqueued() {
    if (host_flag) {
        std::lock_guard<std::mutex> lock(mutex);
        return enqueued;
    } else {
        return true;
    }
}

void Event::mark_enqueued(Stream *stream) {
    if (this->host_flag) {
        std::lock_guard<std::mutex> lock(mutex);
        this->enqueued = true;
    }
#ifdef __CUDACC__
    else {
        cudaSetDevice(device_id);
        cudaEventRecord(this->cuda_event, stream->cuda_stream);
    }
#endif
}

void Event::mark_done() {
    if (host_flag) {
        std::lock_guard<std::mutex> lock(mutex);
        this->enqueued = false;
        cv.notify_all();
    }
}

void Event::wait(Stream *stream) {
    if (this->host_flag) {
        // Host events must be waited on
        std::unique_lock<std::mutex> lock(mutex);
        if (this->enqueued)
            cv.wait(lock, [this](){return not this->enqueued;});
    }
#ifdef __CUDACC__
    else if (stream == nullptr or stream->host_flag) {
        // Hosts wait on CUDA events by synchronizing
        cudaSetDevice(device_id);
        cudaEventSynchronize(cuda_event);
    } else {
        // Devices wait on CUDA events using CUDA API
        cudaSetDevice(device_id);
        cudaStreamWaitEvent(stream->cuda_stream, cuda_event, 0);
    }
#endif
}
