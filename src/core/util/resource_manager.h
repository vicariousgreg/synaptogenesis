#ifndef resource_manager.h
#define resource_manager.h

#include <thread>
#include <vector>
#include <map>

#include "util/parallel.h"
#include "util/constants.h"
#include "util/stream.h"
#include "util/event.h"

class BasePointer;

class ResourceManager {
    public:
        static ResourceManager *get_instance();
        virtual ~ResourceManager();

        /* Delete all managed pointers */
        void flush();

        unsigned int get_num_cores() { return num_cores; }
        unsigned int get_num_devices() { return devices.size(); }
        DeviceID get_host_id() { return devices.size()-1; }
        bool is_host(DeviceID device_id) { return device_id == get_host_id(); }

        void* allocate_host(unsigned long count, int size);
        void* allocate_device(unsigned long count, int size,
            void* source_data, DeviceID device_id=0);

        void transfer(DeviceID device_id, std::vector<BasePointer*> ptrs);

        Stream *get_default_stream(DeviceID id);
        Stream *get_inter_device_stream(DeviceID id);
        Stream *create_stream(DeviceID id);
        Event *create_event(DeviceID id);

    private:
        ResourceManager();
        static ResourceManager *instance;

        class Device {
            public:
                Device(DeviceID device_id, bool host_flag);
                virtual ~Device();

                bool is_host() { return host_flag; }
                Stream *create_stream();
                Event *create_event();

                const DeviceID device_id;
                const bool host_flag;
                Stream* const default_stream;
                Stream* const inter_device_stream;
                std::vector<Stream*> streams;
                std::vector<Event*> events;
        };

        int num_cores;
        std::vector<Device*> devices;
        std::map<DeviceID, std::vector<void*> > managed_pointers;
};

#endif
