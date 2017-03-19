#include "util/stream.h"
#include "util/pointer.h"

Stream::Stream(DeviceID device_id, bool host_flag)
        : device_id(device_id), host_flag(host_flag) {
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaStreamCreate(&this->cuda_stream);
    }
#endif
}

Stream::~Stream() {
#ifdef __CUDACC__
    if (not host_flag and cuda_stream != 0) {
        cudaSetDevice(device_id);
        cudaStreamDestroy(this->cuda_stream);
    }
#endif
}

void Stream::record(Event *event) {
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaEventRecord(event->get_cuda_event(), cuda_stream);
    }
#endif
}

void Stream::wait(Event *event) {
#ifdef __CUDACC__
    if (not host_flag) {
        cudaSetDevice(device_id);
        cudaStreamWaitEvent(cuda_stream, event->get_cuda_event(), 0);
    }
#endif
}

DefaultStream::DefaultStream(DeviceID device_id, bool host_flag) {
    this->device_id = device_id;
    this->host_flag = host_flag;
#ifdef __CUDACC__
    this->cuda_stream = 0;
#endif
}
