#ifndef pointer_h
#define pointer_h

#include "util/parallel.h"
#include "util/stream.h"

class PointerKey {
    public:
        PointerKey(size_t hash, size_t type, size_t bytes, size_t offset)
            : hash(hash), type(type), bytes(bytes), offset(offset) { }
        PointerKey(size_t hash, std::string type, size_t bytes, size_t offset)
            : PointerKey(hash, std::hash<std::string>()(type), bytes, offset) { }
        const size_t hash;
        const size_t type;
        const size_t bytes;
        const size_t offset;

        bool operator <(const PointerKey& other) const {
            if (hash == other.hash) return type < other.type;
            else return hash < other.hash;
        }
        bool operator ==(const PointerKey& other) const {
            return hash == other.hash and type == other.type;
        }
};

class BasePointer {
    public:
        HOST DEVICE void* get(size_t offset=0) { return ptr + (offset * unit_size); }
        HOST DEVICE size_t get_size() { return size; }
        HOST DEVICE size_t get_unit_size() { return unit_size; }
        HOST DEVICE size_t get_bytes() { return size * unit_size; }
        HOST DEVICE DeviceID get_device_id() { return device_id; }

        // Frees the encapsulated pointer if this is the owner
        void free();

        // Transfer data to a device
        void transfer(DeviceID new_device, void* destination,
            bool transfer_ownership);

    protected:
        BasePointer(void* ptr, size_t size, size_t unit_size, DeviceID device_id,
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
        size_t size;
        size_t unit_size;
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
        Pointer(size_t size);
        Pointer(size_t size, T val);

        // Container constructor
        // Encapsulates the pointer but does not claim ownership
        Pointer(T* ptr, size_t size, bool local, DeviceID device_id);


        /*****************************/
        /*** Indirect constructors ***/
        /*****************************/
        // Cast the pointer to a new type
        template<typename S> Pointer<S> cast() const;

        // Slice the pointer, creating a new pointer that represents
        //   a piece of the old pointer with a new size
        Pointer<T> slice(size_t offset, size_t new_size) const;

        // Creates a pinned pointer, which is only supported if CUDA is enabled
        // Comes with a version for initializing
        static Pointer<T> pinned_pointer(size_t size);
        static Pointer<T> pinned_pointer(size_t size, T val);


        /*****************/
        /*** Retrieval ***/
        /*****************/
        operator T*() const { return get(); }

        // Get the encapsulated pointer
        // This method has host/device protections to ensure that pointers
        //   are only accessed from locations where they are relevant
        HOST DEVICE T* get(size_t offset=0) const;

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
