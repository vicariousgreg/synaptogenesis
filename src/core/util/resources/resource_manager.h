#ifndef resource_manager.h
#define resource_manager.h

#include <thread>
#include <vector>
#include <set>
#include <map>

#include "util/parallel.h"
#include "util/constants.h"
#include "util/resources/stream.h"
#include "util/resources/event.h"
#include "util/property_config.h"

class Memstat {
    public:
        Memstat(DeviceID device_id, size_t free, size_t total,
            size_t used, size_t used_by_this);
        void print();
        PropertyConfig to_config();

        DeviceID device_id;
        const size_t free;
        const size_t total;
        const size_t used;
        const size_t used_by_this;
};


class BasePointer;

class ResourceManager {
    public:
        static ResourceManager *get_instance();
        virtual ~ResourceManager();

        /* Delete managed pointers */
        void flush();
        void flush_device();
        void flush_host();
        void flush(DeviceID device_id);

        /* Delete resources */
        void delete_streams();
        void delete_events();

        /* Getters */
        unsigned int get_num_cores() { return num_cores; }
        unsigned int get_num_devices() { return devices.size(); }
        const std::set<DeviceID> get_devices() { return device_ids; }
        const std::set<DeviceID> get_default_devices();
        bool check_device_ids(std::set<DeviceID> ids, bool raise_error=true);
        DeviceID get_host_id() { return devices.size()-1; }
        std::vector<DeviceID> get_gpu_ids();
        std::vector<DeviceID> get_all_ids();
        bool is_host(DeviceID device_id) { return device_id == get_host_id(); }

        /* Memory allocation */
        void* allocate_host(size_t count, size_t size);
        void* allocate_host_pinned(size_t count, size_t size);
        void* allocate_device(size_t count, size_t size,
            void* source_data, DeviceID device_id=0);

        /* Memory usage tracking */
        void drop_pointer(void* ptr, DeviceID device_id);
        std::vector<PropertyConfig> get_memory_usage(bool verbose=false);

        /* Smart pointer count tracking */
        void increment_pointer_count(void* ptr, DeviceID device_id);
        void decrement_pointer_count(void* ptr, DeviceID device_id);

        /* Transfers a set of pointers to a new memory block */
        BasePointer* transfer(DeviceID device_id,
            std::vector<BasePointer*> ptrs);

        /* Stream / event functions */
        Stream *get_default_stream(DeviceID id);
        Stream *get_inter_device_stream(DeviceID id);
        Stream *create_stream(DeviceID id);
        Event *create_event(DeviceID id);
        void remove(Stream* stream);
        void remove(Event* event);

    private:
        static ResourceManager *instance;
        ResourceManager();

        class Device {
            public:
                Device(DeviceID device_id, bool host_flag, bool solo);
                virtual ~Device();

                bool is_host() const { return host_flag; }

                Stream *create_stream();
                Event *create_event();
                void remove(Stream* stream);
                void remove(Event* event);

                void delete_streams();
                void delete_events();

                const DeviceID device_id;
                const bool host_flag;

                Stream* const default_stream;
                Stream* const inter_device_stream;
                std::vector<Stream*> streams;
                std::vector<Event*> events;
        };

        int num_cores;
        std::vector<Device*> devices;
        std::set<DeviceID> device_ids;
        std::map<DeviceID, std::map<void*, size_t>> managed_pointers;
        std::map<DeviceID, std::map<void*, int>> pointer_counts;
        std::map<DeviceID, size_t> memory_usage;
};

#endif
