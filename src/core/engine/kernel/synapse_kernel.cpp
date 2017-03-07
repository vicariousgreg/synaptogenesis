#include "engine/kernel/synapse_kernel.h"

/* Clears input data */
inline void clear_data_SERIAL(Pointer<float> ptr, int count) {
    float* data = ptr.get();

    for (int nid = 0; nid < count; ++nid)
        data[nid] = 0.0;
}

#ifdef __CUDACC__
inline GLOBAL void clear_data_PARALLEL(Pointer<float> ptr, int count) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
        data[nid] = 0.0;
}
Kernel<void(*)(Pointer<float>, int)> get_clear_data() {
    return Kernel<void(*)(Pointer<float>, int)>(
        clear_data_SERIAL, clear_data_PARALLEL);
}
#else
Kernel<void(*)(Pointer<float>, int)> get_clear_data() {
    return Kernel<void(*)(Pointer<float>, int)>(clear_data_SERIAL);
}
#endif



/* Randomizes input data */
inline void randomize_data_SERIAL(Pointer<float> ptr,
        int count, float max, bool init) {
    float* data = ptr.get();

    if (init)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = fRand(0.0, max);
    else
        for (int nid = 0; nid < count; ++nid)
            data[nid] += fRand(0.0, max);
}
#ifdef __CUDACC__
inline GLOBAL void randomize_data_PARALLEL(Pointer<float> ptr,
        int count, float max, bool init) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        float val = curand_uniform(&cuda_rand_states[nid]) * max;
        if (init)
            data[nid] = val;
        else
            data[nid] += val;
    }
}
Kernel<void(*)(Pointer<float>, int, float, bool)> get_randomize_data() {
    return Kernel<void(*)(Pointer<float>, int, float, bool)>(
        randomize_data_SERIAL, randomize_data_PARALLEL);
}
#else
Kernel<void(*)(Pointer<float>, int, float, bool)> get_randomize_data() {
    return Kernel<void(*)(Pointer<float>, int, float, bool)>(
        randomize_data_SERIAL);
}
#endif
