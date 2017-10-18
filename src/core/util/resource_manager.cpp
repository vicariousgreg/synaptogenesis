#include <cstdlib>
#include <ctime>

#include "util/resource_manager.h"
#include "util/error_manager.h"
#include "util/stream.h"
#include "util/event.h"
#include "util/pointer.h"

ResourceManager *ResourceManager::instance = nullptr;

ResourceManager *ResourceManager::get_instance() {
    if (ResourceManager::instance == nullptr) {
        // Take opportunity to seed random generator
        srand(time(nullptr));
        ResourceManager::instance = new ResourceManager();
    }
    return ResourceManager::instance;
}

ResourceManager::ResourceManager() {
    num_cores = std::thread::hardware_concurrency();

    // Create CUDA devices (GPUs)
    for (int i = 0 ; i < get_num_cuda_devices() ; ++i)
        devices.push_back(new Device(devices.size(), false, false));

    // Create host device (CPU)
    devices.push_back(new Device(devices.size(), true, devices.size() == 0));

    // Host or First GPU
    devices[0]->set_active(true);
}

ResourceManager::~ResourceManager() {
    for (auto device : devices) delete device;
    this->flush();
    ResourceManager::instance = nullptr;
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
        devices.at(device_id)->delete_streams();
        devices.at(device_id)->delete_events();
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to flush invalid device ID!");
    }
}

int ResourceManager::get_num_gpus() {
    return get_num_cuda_devices();
}

void ResourceManager::set_cpu() {
    for (auto device : devices)
        device->set_active(false);
    devices[devices.size()-1]->set_active(true);
}

void ResourceManager::set_gpu(int index) {
    if (index > get_num_cuda_devices())
        LOG_ERROR("Could not find GPU " + std::to_string(index));
    for (auto device : devices)
        device->set_active(false);
    devices[index]->set_active(true);
}

void ResourceManager::set_multi_gpu(int num) {
    for (auto device : devices) device->set_active(not device->is_host());
}

void ResourceManager::set_all() {
    for (auto device : devices) device->set_active(true);
}


void* ResourceManager::allocate_host(size_t count, size_t size) {
    if (count == 0) return nullptr;

    void* ptr = calloc(count, size);
    if (ptr == nullptr)
        LOG_ERROR(
            "Failed to allocate space on host for neuron state!");
    managed_pointers[get_host_id()].insert(ptr);
    return ptr;
}

void* ResourceManager::allocate_device(size_t count, size_t size,
        void* source_data, DeviceID device_id) {
    if (count == 0) return nullptr;

    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to allocate memory on non-existent device!");
    void* ptr = cuda_allocate_device(device_id, count, size, source_data);
    managed_pointers[device_id].insert(ptr);
    return ptr;
}

void ResourceManager::drop_pointer(void* ptr, DeviceID device_id) {
    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to drop pointer from non-existent device!");
    managed_pointers[device_id].erase(ptr);
}

BasePointer* ResourceManager::transfer(DeviceID device_id,
        std::vector<BasePointer*> ptrs) {
    char* data = nullptr;
    size_t size = 0;
    for (auto ptr : ptrs)
        size += ptr->get_bytes();

    if (size > 0) {
        if (is_host(device_id))
            data = (char*)this->allocate_host(size, 1);
        else
            data = (char*)this->allocate_device(size, 1, nullptr, device_id);

        char* it = data;
        for (auto ptr : ptrs) {
            if (ptr->size > 0) {
                ptr->transfer(device_id, it, false);
                it += ptr->size * ptr->unit_size;
            }
        }
    }
    return new BasePointer(
        std::type_index(typeid(void)), data, size, 1,
        device_id, is_host(device_id), false, true);
}

Stream *ResourceManager::get_default_stream(DeviceID device_id) {
    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to retrieve default stream for non-existent device!");
    return devices[device_id]->default_stream;
}

Stream *ResourceManager::get_inter_device_stream(DeviceID device_id) {
    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to retrieve inter-device stream"
            " for non-existent device!");
    return devices[device_id]->inter_device_stream;
}

Stream *ResourceManager::create_stream(DeviceID device_id) {
    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to create stream on non-existent device!");
    return devices[device_id]->create_stream();
}

Event *ResourceManager::create_event(DeviceID device_id) {
    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to create event on non-existent device!");
    return devices[device_id]->create_event();
}

void ResourceManager::delete_streams() {
    for (auto device : devices) device->delete_streams();
}

void ResourceManager::delete_events() {
    for (auto device : devices) device->delete_events();
}

const std::vector<DeviceID> ResourceManager::get_active_devices() {
    std::vector<DeviceID> active;
    for (auto device : devices)
        if (device->is_active())
            active.push_back(device->device_id);
    return active;
}


ResourceManager::Device::Device(DeviceID device_id, bool host_flag, bool solo)
        : device_id(device_id),
          host_flag(host_flag),
          active(false),
          default_stream(new DefaultStream(device_id, host_flag)),
          inter_device_stream((solo)
              ? nullptr
              : new Stream(device_id, host_flag)) { }

ResourceManager::Device::~Device() {
    delete default_stream;
    if (inter_device_stream != nullptr) delete inter_device_stream;
    delete_streams();
    delete_events();
}

void ResourceManager::Device::delete_streams() {
    for (auto stream : streams) delete stream;
    streams.clear();
}

void ResourceManager::Device::delete_events() {
    for (auto event : events) delete event;
    events.clear();
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
