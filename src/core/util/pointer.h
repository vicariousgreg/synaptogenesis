#ifndef pointer_h
#define pointer_h

#include <cstdlib>
#include <cstring>

#include "util/parallel.h"
#include "util/tools.h"
#include "util/error_manager.h"

template<class T>
class Pointer {
    public:
        Pointer()
            : ptr(nullptr),
              size(0),
              local(true),
              original(false) { }

        Pointer(int size)
            : ptr((T*)allocate_host(size, sizeof(T))),
              size(size),
              local(true),
              original(true) { }

        Pointer(T* ptr, int size, bool local=true)
            : ptr(ptr),
              size(size),
              local(local),
              original(false) { }

        void free() {
            if (original) {
                if (local) std::free(ptr);
#ifdef PARALLEL
                else cudaFree(this->ptr);
#endif
            }
        }

        void transfer_to_device() {
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


        HOST DEVICE virtual T* get() const {
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

        operator T*() const {
            return this->get();
        }

        template<typename S>
        Pointer<S> cast() {
            return Pointer<S>((S*)ptr, this->size, this->local);
        }

        Pointer<T> splice(int offset, int new_size) const {
            return Pointer<T>(ptr + offset, new_size, this->local);
        }

        void copy_to(T* dst) {
            if (local)
                memcpy(dst, this->ptr, this->size * sizeof(T));
#ifdef PARALLEL
            else
                cudaMemcpyAsync(dst, this->ptr, this->size * sizeof(T),
                    cudaMemcpyDeviceToHost);
#endif
        }

        void copy_from(T* src) {
            if (local)
                memcpy(this->ptr, src, this->size * sizeof(T));
#ifdef PARALLEL
            else
                cudaMemcpyAsync(this->ptr, src, this->size * sizeof(T),
                    cudaMemcpyHostToDevice);
#endif
        }

    protected:
        T* ptr;
        int size;
        bool local;
        bool original;
};

#endif
