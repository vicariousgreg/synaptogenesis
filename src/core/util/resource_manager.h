#ifndef resource_manager.h
#define resource_manager.h

#include <thread>
#include <vector>
#include <map>

#include "util/parallel.h"

typedef unsigned int StreamID;
typedef unsigned int DeviceID;
typedef unsigned int EventID;

class Device {
    public:
        virtual ~Device();

        virtual void create_stream(StreamID id);
        virtual void create_event(EventID id);
        virtual void synchronize_event(EventID id);

    protected:
        friend class ResourceManager;
        Device(DeviceID device_id)
                : device_id(device_id) { }
        const DeviceID device_id;

#ifdef __CUDACC__
        std::map<StreamID, cudaStream_t*> streams;
        std::map<EventID, cudaEvent_t*> events;
#endif
};

class ResourceManager {
    public:
        static ResourceManager *get_instance();
        virtual ~ResourceManager();

        unsigned int get_num_cores() { return num_cores; }
        unsigned int get_num_devices() { return devices.size(); }

        Device *get_device(DeviceID id) { return devices[id]; }

        void* allocate_host(int count, int size);
        void* allocate_device(int count, int size,
            void* source_data, DeviceID device_id=0);

        StreamID create_stream(DeviceID id=0);
        EventID create_event(DeviceID id=0);
        void synchronize_event(EventID id);

    private:
        ResourceManager();
        static ResourceManager *instance;

        int num_cores;
        std::vector<Device*> devices;

        std::vector<DeviceID> stream_devices;
        std::vector<StreamID> default_streams;

        std::vector<DeviceID> event_devices;
};

#endif
