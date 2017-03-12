#ifndef stream_h
#define stream_h

#include "util/event.h"
#include "util/parallel.h"
#include "util/resource_manager.h"

class Stream {
    public:
        Stream(DeviceID device_id, bool host_flag);
        virtual ~Stream();

        void record(Event *event);
        void wait(Event *event);
        bool is_host() { return host_flag; }
        DeviceID get_device_id() { return device_id; }

#ifdef __CUDACC__
        cudaStream_t get_cuda_stream() { return cuda_stream; }
#endif

    protected:
        Stream() { }
        DeviceID device_id;
        bool host_flag;

#ifdef __CUDACC__
        cudaStream_t cuda_stream;
#endif
};

class DefaultStream : public Stream {
    public:
        DefaultStream(DeviceID device_id, bool host_flag);
};

#endif
