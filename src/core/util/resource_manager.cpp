#include <cstdlib>

#include "util/resource_manager.h"
#include "util/error_manager.h"
#include "util/stream.h"
#include "util/event.h"
#include "util/pointer.h"

ResourceManager *ResourceManager::instance = nullptr;

ResourceManager *ResourceManager::get_instance() {
    if (ResourceManager::instance == nullptr)
        ResourceManager::instance = new ResourceManager();
    return ResourceManager::instance;
}

ResourceManager::ResourceManager() {
    num_cores = std::thread::hardware_concurrency();

    // Create CUDA devices (GPUs)
    for (int i = 0 ; i < get_num_cuda_devices() ; ++i)
        devices.push_back(new Device(devices.size(), false));

    // Create host device (CPU)
    devices.push_back(new Device(devices.size(), true));

    for (int i = 0; i < devices.size(); ++i) {
        scheduled_transfers.push_back(std::vector<BasePointer*>());
        scheduled_transfer_size.push_back(0);
    }
}

ResourceManager::~ResourceManager() {
    for (auto device : devices) delete device;
}

void* ResourceManager::allocate_host(unsigned long count, int size) {
    void* ptr = calloc(count, size);
    if (ptr == nullptr)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate space on host for neuron state!");
    return ptr;
}

void* ResourceManager::allocate_device(unsigned long count, int size,
        void* source_data, DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to allocate memory on non-existent device.");
    return cuda_allocate_device(device_id, count, size, source_data);
}

void ResourceManager::schedule_transfer(BasePointer* ptr, DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to allocate memory on non-existent device or host.");
    if (device_id != get_host_id()) {
        scheduled_transfers[device_id].push_back(ptr);
        scheduled_transfer_size[device_id] += ptr->size * ptr->unit_size;
    }
}

void ResourceManager::transfer() {
    for (DeviceID i = 0; i < devices.size(); ++i) {
        unsigned long size = scheduled_transfer_size[i];
        if (size > 0) {
            char* data = (char*)this->allocate_device(size, 1, nullptr, i);
            for (auto ptr : scheduled_transfers[i]) {
                ptr->transfer(i, data);
                data += ptr->size * ptr->unit_size;
            }
        }
    }
}

Stream *ResourceManager::get_default_stream(DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to retrieve default stream for non-existent device.");
    return devices[device_id]->default_stream;
}

Stream *ResourceManager::create_stream(DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to create stream on non-existent device.");
    return devices[device_id]->create_stream();
}

Event *ResourceManager::create_event(DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to create event on non-existent device.");
    return devices[device_id]->create_event();
}

ResourceManager::Device::Device(DeviceID device_id, bool host_flag)
        : device_id(device_id),
          host_flag(host_flag),
          default_stream(new DefaultStream(device_id, host_flag)) { }

ResourceManager::Device::~Device() {
    delete default_stream;
    for (auto stream : streams) delete stream;
    for (auto event : events) delete event;
}

Stream *ResourceManager::Device::create_stream() {
    auto stream = new Stream(device_id, host_flag);
    this->streams.push_back(stream);
    return stream;
}

Event *ResourceManager::Device::create_event() {
    auto event = new Event(device_id, host_flag);
    this->events.push_back(event);
    return event;
}
