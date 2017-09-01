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
}

ResourceManager::~ResourceManager() {
    for (auto device : devices) delete device;
    this->flush();
}

void ResourceManager::flush() {
    for (auto pair : managed_pointers)
        this->flush(pair.first);
}

void ResourceManager::flush_device() {
    for (auto pair : managed_pointers)
        if (not is_host(pair.first))
            this->flush(pair.first);
}

void ResourceManager::flush_host() {
    this->flush(get_host_id());
}

void ResourceManager::flush(DeviceID device_id) {
    try {
        auto pointers = managed_pointers.at(device_id);
        if (is_host(device_id))
            for (auto ptr : pointers)
                std::free(ptr);
#ifdef __CUDACC__
        else {
            cudaSetDevice(device_id);
            for (auto ptr : pointers)
                cudaFree(ptr);
        }
#endif
        managed_pointers.erase(device_id);
    } catch (std::out_of_range) { }
}

void* ResourceManager::allocate_host(size_t count, size_t size) {
    if (count == 0) return nullptr;

    void* ptr = calloc(count, size);
    if (ptr == nullptr)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate space on host for neuron state!");
    return ptr;
}

void* ResourceManager::allocate_device(size_t count, size_t size,
        void* source_data, DeviceID device_id) {
    if (count == 0) return nullptr;

    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to allocate memory on non-existent device.");
    return cuda_allocate_device(device_id, count, size, source_data);
}

void ResourceManager::transfer(DeviceID device_id, std::vector<BasePointer*> ptrs) {
    size_t size = 0;
    for (auto ptr : ptrs)
        size += ptr->size * ptr->unit_size;

    if (size > 0) {
        char* data;

        if (is_host(device_id))
            data = (char*)this->allocate_host(size, 1);
        else
            data = (char*)this->allocate_device(size, 1, nullptr, device_id);

        // Maintain ownership of block
        // Pointers cannot free it, so the Resource Manager has to handle it
        this->managed_pointers[device_id].push_back(data);

        for (auto ptr : ptrs) {
            if (ptr->size > 0) {
                ptr->transfer(device_id, data, false);
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

Stream *ResourceManager::get_inter_device_stream(DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to retrieve inter-device stream for non-existent device.");
    return devices[device_id]->inter_device_stream;
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
          default_stream(new DefaultStream(device_id, host_flag)),
          inter_device_stream(new Stream(device_id, host_flag)) { }

ResourceManager::Device::~Device() {
    delete default_stream;
    delete inter_device_stream;
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
