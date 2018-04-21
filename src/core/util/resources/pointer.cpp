#ifndef pointer_cpp
#define pointer_cpp

#ifndef pointer_h
#include "util/resources/pointer.h"
#endif

#include <cstdlib>
#include <cstring>
#include <assert.h>

#include "util/logger.h"
#include "util/resources/resource_manager.h"

template<typename T>
Pointer<T>::Pointer()
    : BasePointer(
          std::type_index(typeid(T)),
          nullptr,
          0,
          sizeof(T),
          ResourceManager::get_instance()->get_host_id(),
          false) { }

template<typename T>
Pointer<T>::Pointer(size_t size)
    : BasePointer(
          std::type_index(typeid(T)),
          ResourceManager::get_instance()->allocate_host(size, sizeof(T)),
          size,
          sizeof(T),
          ResourceManager::get_instance()->get_host_id(),
          true) { }

template<typename T>
Pointer<T>::Pointer(size_t size, T val)
        : Pointer<T>(size) {
    this->set(val, false);
}

template<typename T>
Pointer<T>::Pointer(T* ptr, size_t size, DeviceID device_id,
        bool claim_ownership)
    : BasePointer(
          std::type_index(typeid(T)),
          ptr,
          size,
          sizeof(T),
          device_id,
          claim_ownership) { }

template<typename T>
Pointer<T>::Pointer(BasePointer* base_ptr)
    : BasePointer(
          std::type_index(typeid(T)),
          (float*)base_ptr->get(),
          base_ptr->get_size(),
          sizeof(T),
          base_ptr->get_device_id(),
          false) { }

template<typename T>
Pointer<T>::Pointer(const Pointer<T>& other)
    : BasePointer(
          std::type_index(typeid(T)),
          other.ptr,
          other.size,
          sizeof(T),
          other.device_id,
          false) {
    this->pinned = other.pinned;
}

template<typename T>
Pointer<T>::Pointer(Pointer<T>& other, bool claim_ownership)
    : BasePointer(
          std::type_index(typeid(T)),
          other.ptr,
          other.size,
          sizeof(T),
          other.device_id,
          claim_ownership) {
    this->pinned = other.pinned;
    if (claim_ownership) {
        if (not other.owner)
            LOG_ERROR("Attempted to pass ownership from non-owner pointer!");
        other.owner = false;
        other.ptr = nullptr;
        other.size = 0;
    }
}

template<typename T>
template<typename S>
Pointer<S> Pointer<T>::cast() const {
    return Pointer<S>((S*)ptr, size, device_id);
}

template<typename T>
Pointer<T> Pointer<T>::slice(size_t offset, size_t new_size) const {
    return Pointer<T>(((T*)ptr) + offset, new_size, device_id);
}

template<typename T>
Pointer<T> Pointer<T>::pinned_pointer(size_t size) {
#ifdef __CUDACC__
    T* ptr = (T*)ResourceManager::get_instance()
        ->allocate_host_pinned(size, sizeof(T));
    auto pointer = Pointer<T>(ptr, size,
        ResourceManager::get_instance()->get_host_id(),
        true);
    pointer.pinned = true;
    return pointer;
#else
    return Pointer<T>(size);
#endif
}

template<typename T>
Pointer<T> Pointer<T>::pinned_pointer(size_t size, T val) {
    auto pointer = Pointer<T>::pinned_pointer(size);
    pointer.set(val, false);
    return pointer;
}

template<typename T>
Pointer<T> Pointer<T>::device_pointer(DeviceID device_id, size_t size) {
    auto res_man = ResourceManager::get_instance();
    T* ptr = (device_id == res_man->get_host_id())
        ? (T*)res_man->allocate_host(size, sizeof(T))
        : (T*)res_man->allocate_device(
            size, sizeof(T), nullptr, device_id);
    return Pointer<T>(ptr, size, device_id, true);
}

template<typename T>
Pointer<T> Pointer<T>::device_pointer(DeviceID device_id, size_t size, T val) {
    auto pointer = Pointer<T>::device_pointer(device_id, size);
    pointer.set(val, false);
    return pointer;
}

template<typename T>
HOST DEVICE T* Pointer<T>::get(size_t offset) const {
    if (size == 0) return nullptr;
#ifdef __CUDA_ARCH__
    if (offset >= size) assert(false);
    if (local and ptr != nullptr) assert(false);
    return ((T*)ptr) + offset;
#else
    if (offset >= size)
        LOG_ERROR(
            "Attempted to dereference pointer with invalid offset!");
    if (not local and ptr != nullptr)
        LOG_ERROR(
            "Attempted to dereference device pointer from host!");
    return ((T*)ptr) + offset;
#endif
}

template<typename T>
HOST DEVICE T* Pointer<T>::get_unsafe(size_t offset) const {
    if (size == 0) return nullptr;
    if (offset >= size)
        LOG_ERROR(
            "Attempted to dereference pointer with invalid offset!");
    return ((T*)ptr) + offset;
}

template<typename T>
void Pointer<T>::copy_to(Pointer<T> dst) const {
    if (dst.size != this->size)
        LOG_ERROR(
            "Attempted to copy memory between pointers of different sizes!");
    if (local and dst.local)
        memcpy(dst.ptr, ptr, size * sizeof(T));
    else
        LOG_ERROR(
            "Non-local transfers must be handled by a Stream!");
}

template<typename T>
void Pointer<T>::copy_to(Pointer<T> dst, Stream *stream) const {
    if (dst.size != this->size)
        LOG_ERROR(
            "Attempted to copy memory between pointers of different sizes!");
#ifdef __CUDACC__
    if (this->local and dst.local)
        memcpy(dst.ptr, this->ptr, this->size * sizeof(T));
    else {
        if (stream->is_host() and not this->local)
            LOG_ERROR(
                "Attempted to copy memory between devices using host stream!");

        if (stream->is_host())
            cudaSetDevice(dst.get_device_id());
        else
            cudaSetDevice(stream->get_device_id());

        if (not this->local and not dst.local) {
            cudaMemcpyPeerAsync(dst.ptr, dst.device_id,
                this->ptr, this->device_id, this->size * sizeof(T),
                stream->get_cuda_stream());
        } else {
            if (this->local)
                cudaMemcpyAsync(dst.ptr, this->ptr, this->size * sizeof(T),
                    cudaMemcpyHostToDevice, stream->get_cuda_stream());
            else
                cudaMemcpyAsync(dst.ptr, this->ptr, this->size * sizeof(T),
                    cudaMemcpyDeviceToHost, stream->get_cuda_stream());

        }
    }
#else
    memcpy(dst.ptr, this->ptr, this->size * sizeof(T));
#endif
}

template<typename T>
void Pointer<T>::set(T val, bool async) {
    T* t_ptr = (T*)ptr;
    if (local) {
        for (size_t i = 0 ; i < size ; ++i) t_ptr[i] = val;
#ifdef __CUDACC__
    } else if (sizeof(T) == 1) {
        cudaSetDevice(device_id);
        if (async) cudaMemsetAsync(ptr,val,size);
        else cudaMemset(ptr,val,size);
    } else if (sizeof(T) == 4) {
        cudaSetDevice(device_id);
        if (async) cuMemsetD32Async((CUdeviceptr)ptr,val,size, 0);
        else cuMemsetD32((CUdeviceptr)ptr,val,size);
    } else {
        LOG_ERROR(
            "Attempted to set memory of non-primitive device array!");
#endif
    }
}

#endif
