#include <cstdlib>
#include <ctime>
#include <sys/sysinfo.h>

#include "util/resource_manager.h"
#include "util/error_manager.h"
#include "util/stream.h"
#include "util/event.h"
#include "util/pointer.h"

Memstat::Memstat(DeviceID device_id, size_t free, size_t total,
    size_t used, size_t used_by_this)
        : device_id(device_id), free(free), total(total),
          used(used), used_by_this(used_by_this) { }

void Memstat::print() {
    if (free == 0)
        printf("Device %d memory usage:\n"
            "  used  : %11zu   %8.2f MB\n",
            device_id,
            used_by_this, float(used_by_this)/1024.0/1024.0);
    else if (used_by_this > 0)
        printf("Device %d memory usage:\n"
            "  proc  : %11zu   %8.2f MB\n"
            "  used  : %11zu   %8.2f MB\n"
            "  free  : %11zu   %8.2f MB\n"
            "  total : %11zu   %8.2f MB\n",
            device_id,
            used_by_this, float(used_by_this)/1024.0/1024.0,
            used, float(used)/1024.0/1024.0,
            free, float(free)/1024.0/1024.0,
            total, float(total)/1024.0/1024.0);
    else
        printf("Device %d memory usage:\n"
            "  used  : %11zu   %8.2f MB\n"
            "  free  : %11zu   %8.2f MB\n"
            "  total : %11zu   %8.2f MB\n",
            device_id,
            used, float(used)/1024.0/1024.0,
            free, float(free)/1024.0/1024.0,
            total, float(total)/1024.0/1024.0);
}

PropertyConfig Memstat::to_config() {
    auto host_id = ResourceManager::get_instance()->get_host_id();
    return PropertyConfig({
        {"device id", std::to_string(device_id)},
        {"device type", (device_id == host_id) ? "host" : "gpu"},
        {"proc", std::to_string(float(used_by_this)/1024.0/1024.0)},
        {"used", std::to_string(float(used)/1024.0/1024.0)},
        {"free", std::to_string(float(free)/1024.0/1024.0)},
        {"total", std::to_string(float(total)/1024.0/1024.0)} });
}


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

    // Create DeviceID set
    for (auto d : devices) device_ids.insert(d->device_id);

    // Initialize memory usage
    for (auto d : devices)
        memory_usage[d->device_id] = 0;
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
                std::free(ptr.first);
#ifdef __CUDACC__
        else {
            cudaSetDevice(device_id);
            for (auto ptr : pointers)
                cudaFree(ptr.first);
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

const std::set<DeviceID> ResourceManager::get_default_devices() {
    return { devices[0]->device_id };
}

bool ResourceManager::check_device_ids(std::set<DeviceID> ids, bool raise_error) {
    for (auto id : ids)
        if (device_ids.count(id) == 0)
            if (raise_error)
                LOG_ERROR("Invalid device ID: " + std::to_string(id));
            else return false;
    return true;
}

std::vector<DeviceID> ResourceManager::get_gpu_ids() {
    std::vector<DeviceID> gpu_ids;
    for (auto id : device_ids)
        if (id != get_host_id())
            gpu_ids.push_back(id);
    return gpu_ids;
}

std::vector<DeviceID> ResourceManager::get_all_ids() {
    std::vector<DeviceID> all_ids;
    for (auto id : device_ids) all_ids.push_back(id);
    return all_ids;
}

void* ResourceManager::allocate_host(size_t count, size_t size) {
    if (count == 0) return nullptr;

    void* ptr = calloc(count, size);
    if (ptr == nullptr)
        LOG_ERROR(
            "Failed to allocate space on host for neuron state!");
    managed_pointers[get_host_id()].insert({ ptr, count * size });
    this->memory_usage[get_host_id()] += count * size;
    return ptr;
}

void* ResourceManager::allocate_device(size_t count, size_t size,
        void* source_data, DeviceID device_id) {
    if (count == 0) return nullptr;

    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to allocate memory on non-existent device!");
    void* ptr = cuda_allocate_device(device_id, count, size, source_data);
    managed_pointers[device_id].insert({ ptr, count * size });
    this->memory_usage[device_id] += count * size;
    return ptr;
}

void ResourceManager::drop_pointer(void* ptr, size_t bytes, DeviceID device_id) {
    if (device_id >= get_num_devices())
        LOG_ERROR(
            "Attempted to drop pointer from non-existent device!");
    managed_pointers[device_id].erase({ ptr, bytes });
    this->memory_usage[device_id] -= bytes;
}

std::vector<PropertyConfig> ResourceManager::get_memory_usage(bool verbose) {
    std::vector<PropertyConfig> stats;
    for (auto id : device_ids) {
        size_t free, total;
        if (id == get_host_id()) {
            struct sysinfo info;
            sysinfo(&info);
            free = info.freeram;
            total = info.totalram;
        } else {
            device_check_memory(id, &free, &total);
        }
        Memstat stat = Memstat(
            id, free, total, total-free, memory_usage[id]);
        if (verbose) stat.print();
        stats.push_back(stat.to_config());
    }
    return stats;
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
                it += ptr->get_bytes();
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

ResourceManager::Device::Device(DeviceID device_id, bool host_flag, bool solo)
        : device_id(device_id),
          host_flag(host_flag),
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
