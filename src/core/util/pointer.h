#ifndef pointer_h
#define pointer_h

#include "util/parallel.h"

template<class T>
class Pointer {
    public:
        /********************/
        /*** Constructors ***/
        /********************/
        // Null pointer constructor
        Pointer();

        // Allocator constructor
        // Allocates space and claims ownership of pointer
        // Second version takes an initialization value
        Pointer(int size);
        Pointer(int size, T val);

        // Container constructor
        // Encapsulates the pointer but does not claim ownership
        Pointer(T* ptr, int size, bool local=true);


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
        static Pointer<T> pinned_pointer(int size);
        static Pointer<T> pinned_pointer(int size, T val);


        /*****************/
        /*** Retrieval ***/
        /*****************/
        operator T*() const { return get(); }

        // Get the encapsulated pointer
        // This method has host/device protections to ensure that pointers
        //   are only accessed from locations where they are relevant
        HOST DEVICE T* get(int offset=0) const;


        /*************************/
        /*** Memory management ***/
        /*************************/
        // Frees the encapsulated pointer if this is the owner
        void free();

        // Transfer the data to the device
        void transfer_to_device();

        // Copy data from this pointer to a given destination
        void copy_to(Pointer<T> dst, bool async=true) const;

        // Sets memory
        void set(T val, bool async=true);

    protected:
        T* ptr;
        int size;
        bool local;
        bool pinned;
        bool owner;
};

#ifndef pointer_cpp
#include "util/pointer.cpp"
#endif

#endif
