#include <cstdlib>
#include <cstring>

#include "util/pointer.h"
#include "util/error_manager.h"
#include "util/resource_manager.h"
#include "util/tools.h"

void BasePointer::free() {
    if (owner and size > 0) {
#ifdef __CUDACC__
        if (local) {
            if (pinned) cudaFreeHost(ptr); // cuda pinned memory
            else std::free(ptr);           // unpinned host memory
        } else {
            cudaFree(this->ptr);           // cuda device memory
        }
#else
        if (local) std::free(ptr);         // unpinned host memory (default)
#endif
    }
}

void BasePointer::schedule_transfer(DeviceID device_id) {
#ifdef __CUDACC__
    ResourceManager::get_instance()->schedule_transfer(this, device_id);
#endif
}

void BasePointer::transfer(DeviceID device_id, void* destination) {
#ifdef __CUDACC__
    if (local) {
        cudaSetDevice(device_id);
        cudaMemcpy(destination, this->ptr, this->size * this->unit_size,
            cudaMemcpyHostToDevice);
        std::free(this->ptr);
        this->ptr = destination;
        this->local = false;
        this->device_id = device_id;
    } else {
        ErrorManager::get_instance()->log_error(
            "Attempted to transfer device pointer to device!");
    }
#endif
}
