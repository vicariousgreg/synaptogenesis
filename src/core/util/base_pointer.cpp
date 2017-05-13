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

void BasePointer::transfer(DeviceID new_device, void* destination,
        bool transfer_ownership) {
#ifdef __CUDACC__
    bool host_dest = ResourceManager::get_instance()->is_host(new_device);

    if (local) {
        if (host_dest)
            ErrorManager::get_instance()->log_error(
                "Attempted to transfer host pointer to host!");

        cudaSetDevice(new_device);
        cudaMemcpy(destination, this->ptr, this->size * this->unit_size,
            cudaMemcpyHostToDevice);

        if (this->owner) std::free(this->ptr);
        this->ptr = destination;
        this->local = false;
        this->owner = transfer_ownership;
        this->device_id = new_device;
    } else {
        if (not host_dest)
            ErrorManager::get_instance()->log_error(
                "Attempted to transfer device pointer to device!");

        cudaSetDevice(this->device_id);
        cudaMemcpy(destination, this->ptr, this->size * this->unit_size,
            cudaMemcpyDeviceToHost);

        if (this->owner) cudaFree(this->ptr);
        this->ptr = destination;
        this->local = true;
        this->owner = transfer_ownership;
        this->device_id = new_device;
    }
#endif
}
