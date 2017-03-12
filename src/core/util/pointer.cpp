#ifndef pointer_cpp
#define pointer_cpp

#ifndef pointer_h
#include "util/pointer.h"
#endif

#include <cstdlib>
#include <cstring>

#include "util/error_manager.h"
#include "util/resource_manager.h"
#include "util/tools.h"

template<typename T>
Pointer<T>::Pointer()
    : ptr(nullptr),
      size(0),
      local(true),
      pinned(false),
      owner(false) { }

template<typename T>
Pointer<T>::Pointer(int size)
    : ptr((T*)ResourceManager::get_instance()->allocate_host(size, sizeof(T))),
      size(size),
      local(true),
      pinned(false),
      owner(true) { }

template<typename T>
Pointer<T>::Pointer(int size, T val)
        : Pointer<T>(size) {
    this->set(val, false);
}

template<typename T>
Pointer<T>::Pointer(T* ptr, int size, bool local)
    : ptr(ptr),
      size(size),
      local(local),
      pinned(false),
      owner(false) { }

template<typename T>
template<typename S>
Pointer<S> Pointer<T>::cast() const {
    return Pointer<S>((S*)ptr, size, local);
}

template<typename T>
Pointer<T> Pointer<T>::slice(int offset, int new_size) const {
    return Pointer<T>(ptr + offset, new_size, local);
}

template<typename T>
Pointer<T> Pointer<T>::pinned_pointer(int size) {
#ifdef __CUDACC__
    T* ptr;
    cudaMallocHost((void**) &ptr, size * sizeof(T));
    auto pointer = Pointer<T>(ptr, size);
    pointer.owner = true;
    pointer.pinned = true;
    return pointer;
#else
    return Pointer<T>(size);
#endif
}

template<typename T>
Pointer<T> Pointer<T>::pinned_pointer(int size, T val) {
    auto pointer = Pointer<T>::pinned_pointer(size);
    pointer.set(val, false);
    return pointer;
}

template<typename T>
HOST DEVICE T* Pointer<T>::get(int offset) const {
#ifdef __CUDA_ARCH__
    if (local) assert(false);
    return ptr + offset;
#else
    if (not local)
        ErrorManager::get_instance()->log_error(
            "Attempted to dereference device pointer from host!");
    return ptr + offset;
#endif
}

template<typename T>
void Pointer<T>::free() {
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

template<typename T>
void Pointer<T>::transfer_to_device(int device_id) {
#ifdef __CUDACC__
    if (local) {
        T* new_ptr = (T*) ResourceManager::get_instance()->allocate_device(
            size, sizeof(T), this->ptr, device_id);
        std::free(this->ptr);
        this->ptr = new_ptr;
        local = false;
    } else {
        ErrorManager::get_instance()->log_error(
            "Attempted to transfer device pointer to device!");
    }
#endif
}

template<typename T>
void Pointer<T>::copy_to(Pointer<T> dst) const {
    if (dst.size != this->size)
        ErrorManager::get_instance()->log_error(
            "Attempted to copy memory between pointers of different sizes!");
    if (local and dst.local)
        memcpy(dst.ptr, ptr, size * sizeof(T));
    else
        ErrorManager::get_instance()->log_error(
            "Non-local transfers must be handled by a Stream!");
}

template<typename T>
void Pointer<T>::copy_to(Pointer<T> dst, Stream *stream) const {
    if (dst.size != this->size)
        ErrorManager::get_instance()->log_error(
            "Attempted to copy memory between pointers of different sizes!");
#ifdef __CUDACC__
    if (this->local and dst.local) memcpy(dst.ptr, this->ptr, this->size * sizeof(T));
    else {
        auto kind = cudaMemcpyDeviceToDevice;

        if (this->local and not dst.local) kind = cudaMemcpyDeviceToHost;
        else if (dst.local) kind = cudaMemcpyHostToDevice;

        cudaMemcpyAsync(dst.ptr, this->ptr, this->size * sizeof(T),
            kind, stream->cuda_stream);
    }
#else
    memcpy(dst.ptr, this->ptr, this->size * sizeof(T));
#endif
}

template<typename T>
void Pointer<T>::set(T val, bool async) {
    if (local) {
        for (int i = 0 ; i < size ; ++i) ptr[i] = val;
#ifdef __CUDACC__
    } else if (sizeof(T) == 1) {
        if (async) cudaMemsetAsync(ptr,val,size);
        else cudaMemset(ptr,val,size);
    } else if (sizeof(T) == 4) {
        if (async) cuMemsetD32Async((CUdeviceptr)ptr,val,size, 0);
        else cuMemsetD32((CUdeviceptr)ptr,val,size);
    } else {
        ErrorManager::get_instance()->log_error(
            "Attempted to set memory of non-primitive device array!");
#endif
    }
}

#endif
