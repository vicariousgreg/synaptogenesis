#include <cstdlib>

#include "util/resource_manager.h"
#include "util/error_manager.h"
#include "util/stream.h"
#include "util/event.h"

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
}

ResourceManager::~ResourceManager() {
    for (auto device : devices) delete device;
}

void* ResourceManager::allocate_host(int count, int size) {
    void* ptr = calloc(count, size);
    if (ptr == nullptr)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate space on host for neuron state!");
    return ptr;
}

void* ResourceManager::allocate_device(int count, int size,
        void* source_data, DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to allocate memory on non-existent device.");
    return cuda_allocate_device(device_id, count, size, source_data);
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
    auto event = new Event();
    this->events.push_back(event);
    return event;
}
