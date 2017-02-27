#include "engine/stream/stream.h"

Stream::Stream() {
#ifdef __CUDACC__
    this->default_stream = false;
    cudaStreamCreate(&this->cuda_stream);
#endif
}

Stream::Stream(bool default_stream)
    : default_stream(default_stream) { }

Stream::~Stream() {
#ifdef __CUDACC__
    if (not default_stream)
        cudaStreamDestroy(this->cuda_stream);
#endif
}

void Stream::record(Event *event) {
#ifdef __CUDACC__
    cudaEventRecord(event->cuda_event, cuda_stream);
#endif
}

void Stream::wait(Event *event) {
#ifdef __CUDACC__
    cudaStreamWaitEvent(cuda_stream, event->cuda_event, 0);
#endif
}

void DefaultStream::record(Event *event) {
#ifdef __CUDACC__
    cudaEventRecord(event->cuda_event, 0);
#endif
}

void DefaultStream::wait(Event *event) {
#ifdef __CUDACC__
    cudaStreamWaitEvent(0, event->cuda_event, 0);
#endif
}
