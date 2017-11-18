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
    if (destination == nullptr) {
        transfer_ownership = true;
        destination = (void*)ResourceManager::get_instance()->allocate_device(
            size, unit_size, nullptr, new_device);
    }

    if (local) {
        if (host_dest)
            memcpy(destination, this->ptr, this->size * this->unit_size);
        else {
            cudaSetDevice(new_device);
            cudaMemcpy(destination, this->ptr, this->size * this->unit_size,
                cudaMemcpyHostToDevice);
        }
    } else {
        cudaSetDevice(this->device_id);
        if (host_dest) {
            cudaMemcpy(destination, this->ptr, this->size * this->unit_size,
                cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpyPeer(destination, new_device,
                this->ptr, this->device_id,
                this->size * this->unit_size);
            device_synchronize();
        }
    }
    if (this->owner) this->free();
    this->ptr = destination;
    this->owner = transfer_ownership;
    this->device_id = new_device;
    this->local = host_dest;
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
