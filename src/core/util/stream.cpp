#include "util/stream.h"
#include "util/pointer.h"

Stream::Stream() {
#ifdef __CUDACC__
    cudaStreamCreate(&this->cuda_stream);
#endif
}

Stream::~Stream() {
#ifdef __CUDACC__
    if (cuda_stream != 0)
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

DefaultStream::DefaultStream() {
#ifdef __CUDACC__
    this->cuda_stream = 0;
#endif
}
