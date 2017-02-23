#ifndef pointer_cpp
#define pointer_cpp

#ifndef pointer_h
#include "util/pointer.h"
#endif

#include <cstdlib>
#include <cstring>

#include "util/error_manager.h"
#include "util/tools.h"

template<typename T>
Pointer<T>::Pointer()
    : ptr(nullptr),
      size(0),
      local(true),
      owner(false) { }

template<typename T>
Pointer<T>::Pointer(int size)
    : ptr((T*)allocate_host(size, sizeof(T))),
      size(size),
      local(true),
      owner(true) { }

template<typename T>
Pointer<T>::Pointer(T* ptr, int size, bool local)
    : ptr(ptr),
      size(size),
      local(local),
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
HOST DEVICE T* Pointer<T>::get() const {
#ifdef __CUDA_ARCH__
    if (local) assert(false);
    return ptr;
#else
    if (not local)
        ErrorManager::get_instance()->log_error(
            "Attempted to dereference device pointer from host!");
    return ptr;
#endif
}

template<typename T>
void Pointer<T>::free() {
    if (owner) {
        if (local) std::free(ptr);
#ifdef PARALLEL
        else cudaFree(this->ptr);
#endif
    }
}

template<typename T>
void Pointer<T>::transfer_to_device() {
#ifdef PARALLEL
    if (local) {
        T* new_ptr = (T*) allocate_device(size, sizeof(T), this->ptr);
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
void Pointer<T>::copy_to(T* dst) const {
    if (local) memcpy(dst, ptr, size * sizeof(T));
#ifdef PARALLEL
    else cudaMemcpyAsync(dst, ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
#endif
}

template<typename T>
void Pointer<T>::copy_from(T* src) {
    if (local) memcpy(ptr, src, size * sizeof(T));
#ifdef PARALLEL
    else cudaMemcpyAsync(ptr, src, size * sizeof(T), cudaMemcpyHostToDevice);
#endif
}

#endif
