#ifndef resource_manager.h
#define resource_manager.h

#include <thread>
#include <vector>

#include "util/parallel.h"

class Stream;
class Event;

typedef unsigned int DeviceID;

class ResourceManager {
        class Device;

    public:
        static ResourceManager *get_instance();
        virtual ~ResourceManager();

        unsigned int get_num_cores() { return num_cores; }
        unsigned int get_num_devices() { return devices.size(); }

        Device *get_device(DeviceID id) { return devices[id]; }

        void* allocate_host(int count, int size);
        void* allocate_device(int count, int size,
            void* source_data, DeviceID device_id=0);

        Stream *get_default_stream(DeviceID id=0);
        Stream *create_stream(DeviceID id=0);
        Event *create_event(DeviceID id=0);

    private:
        ResourceManager();
        static ResourceManager *instance;

        int num_cores;
        std::vector<Device*> devices;

        class Device {
            public:
                Device(DeviceID device_id);
                virtual ~Device();

                Stream *create_stream();
                Event *create_event();

                const DeviceID device_id;
                Stream* const default_stream;
                std::vector<Stream*> streams;
                std::vector<Event*> events;
        };
};

#endif
