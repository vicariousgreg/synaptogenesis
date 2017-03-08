#include "engine/stream/stream.h"

Stream *Stream::default_stream = 0;

Stream *Stream::get_default_stream() {
    if (Stream::default_stream == nullptr)
        Stream::default_stream = new Stream(true);
    return Stream::default_stream;
}

Stream::Stream() {
    this->is_default_stream = false;
#ifdef __CUDACC__
    cudaStreamCreate(&this->cuda_stream);
#endif
}

Stream::Stream(bool is_default_stream)
    : is_default_stream(is_default_stream) { }

Stream::~Stream() {
#ifdef __CUDACC__
    if (not is_default_stream)
        cudaStreamDestroy(this->cuda_stream);
#endif
}

void Stream::record(Event *event) {
#ifdef __CUDACC__
    if (is_default_stream)
        cudaEventRecord(event->cuda_event, 0);
    else
        cudaEventRecord(event->cuda_event, cuda_stream);
#endif
}

void Stream::wait(Event *event) {
#ifdef __CUDACC__
    if (is_default_stream)
        cudaStreamWaitEvent(0, event->cuda_event, 0);
    else
        cudaStreamWaitEvent(cuda_stream, event->cuda_event, 0);
#endif
}
