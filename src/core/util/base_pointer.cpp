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
            device_check_error(nullptr);
        }
#else
        if (local) std::free(ptr);         // unpinned host memory (default)
#endif
        ResourceManager::get_instance()->drop_pointer(ptr, device_id);
    }
}

BasePointer* BasePointer::slice(size_t offset, size_t new_size) const {
    return new BasePointer(
        type,
        ((char*)ptr) + (offset * unit_size),
        new_size,
        unit_size,
        device_id,
        local,
        pinned,
        false);
}

void BasePointer::transfer(DeviceID new_device, void* destination,
        bool transfer_ownership) {
#ifdef __CUDACC__
    bool host_dest = ResourceManager::get_instance()->is_host(new_device);

    if (local) {
        if (host_dest)
            LOG_ERROR(
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
            LOG_ERROR(
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

void BasePointer::copy_to(BasePointer* other) {
    if (other->size != this->size or other->unit_size != this->unit_size)
        LOG_ERROR(
            "Attempted to copy memory between pointers of different sizes!");

    if (this->local and other->local)
        memcpy(other->ptr, this->ptr, this->size * this->unit_size);
    else
        LOG_ERROR(
            "Attempted to copy memory between base pointers "
            "that aren't on the host!");
}
