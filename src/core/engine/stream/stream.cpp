#include "engine/stream/stream.h"

Stream *Stream::default_stream = 0;

Stream *Stream::get_default_stream() {
    if (Stream::default_stream == nullptr)
        Stream::default_stream = new DefaultStream();
    return Stream::default_stream;
}

Stream::Stream() {
#ifdef __CUDACC__
    this->default_stream = false;
    cudaStreamCreate(&this->cuda_stream);
#endif
}

Stream::Stream(bool is_default_stream)
    : is_default_stream(is_default_stream) { }

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
