#ifndef pointer_h
#define pointer_h

#include "util/parallel.h"
#include "util/stream.h"

class BasePointer {
    public:
        int get_size() { return size; }
        DeviceID get_device_id() { return device_id; }

        // Frees the encapsulated pointer if this is the owner
        void free();

        // Transfer data to a device
        void transfer(DeviceID new_device, void* destination,
            bool transfer_ownership);

    protected:
        BasePointer(void* ptr, unsigned long size, int unit_size, DeviceID device_id,
            bool local, bool pinned, bool owner)
                : ptr(ptr),
                  size(size),
                  unit_size(unit_size),
                  device_id(device_id),
                  local(local),
                  pinned(pinned),
                  owner(owner) { }

        friend class ResourceManager;

        void* ptr;
        unsigned long size;
        int unit_size;
        DeviceID device_id;
        bool local;
        bool pinned;
        bool owner;
};

template<class T>
class Pointer : public BasePointer {
    public:
        /********************/
        /*** Constructors ***/
        /********************/
        // Null pointer constructor
        Pointer();

        // Allocator constructor
        // Allocates space and claims ownership of pointer
        // Second version takes an initialization value
        Pointer(unsigned long size);
        Pointer(unsigned long size, T val);

        // Container constructor
        // Encapsulates the pointer but does not claim ownership
        Pointer(T* ptr, unsigned long size, bool local, DeviceID device_id);


        /*****************************/
        /*** Indirect constructors ***/
        /*****************************/
        // Cast the pointer to a new type
        template<typename S> Pointer<S> cast() const;

        // Slice the pointer, creating a new pointer that represents
        //   a piece of the old pointer with a new size
        Pointer<T> slice(int offset, int new_size) const;

        // Creates a pinned pointer, which is only supported if CUDA is enabled
        // Comes with a version for initializing
        static Pointer<T> pinned_pointer(unsigned long size);
        static Pointer<T> pinned_pointer(unsigned long size, T val);


        /*****************/
        /*** Retrieval ***/
        /*****************/
        operator T*() const { return get(); }

        // Get the encapsulated pointer
        // This method has host/device protections to ensure that pointers
        //   are only accessed from locations where they are relevant
        HOST DEVICE T* get(int offset=0) const;

        // Retrieve containing device ID
        DeviceID get_device_id() { return device_id; }


        bool operator==(const Pointer<T> &other) const {
            return ptr == other.ptr
                and size == other.size
                and device_id == other.device_id;
        }

        bool operator!=(const Pointer<T> &other) const { !(*this == other); }



        /*************************/
        /*** Memory management ***/
        /*************************/
        // Copy data from this pointer to a given destination
        void copy_to(Pointer<T> dst) const;
        void copy_to(Pointer<T> dst, Stream *stream) const;

        // Sets memory
        void set(T val, bool async=true);
};

#ifndef pointer_cpp
#include "util/pointer.cpp"
#endif

#endif
