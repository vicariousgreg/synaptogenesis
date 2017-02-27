#include "engine/stream/event.h"

Event::Event() {
#ifdef __CUDACC__
    cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);
#endif
}

Event::~Event() {
#ifdef __CUDACC__
    cudaEventDestroy(cuda_event);
#endif
}

void Event::synchronize() {
#ifdef __CUDACC__
    cudaEventSynchronize(cuda_event);
#endif
}
