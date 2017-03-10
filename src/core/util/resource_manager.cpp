#include <cstdlib>

#include "util/resource_manager.h"
#include "util/error_manager.h"

Device::~Device() {
#ifdef __CUDACC__
    for (auto pair : streams) {
        cudaStreamDestroy(*pair.second);
        delete pair.second;
    }

    for (auto pair : events) {
        cudaEventDestroy(*pair.second);
        delete pair.second;
    }
#endif
}

void Device::create_stream(StreamID id) {
#ifdef __CUDACC__
    auto cuda_stream = new cudaStream_t;
    this->streams[id] = cuda_stream;
    cudaStreamCreate(cuda_stream);
#endif
}

void Device::create_event(EventID id) {
#ifdef __CUDACC__
    auto cuda_event = new cudaEvent_t;
    this->events[id] = cuda_event;
    cudaEventCreateWithFlags(cuda_event, cudaEventDisableTiming);
#endif
}

void Device::synchronize_event(EventID id) {
#ifdef __CUDACC__
    cudaEventSynchronize(*events[id]);
#endif
}

ResourceManager *ResourceManager::instance = nullptr;

ResourceManager *ResourceManager::get_instance() {
    if (ResourceManager::instance == nullptr)
        ResourceManager::instance = new ResourceManager();
    return ResourceManager::instance;
}

ResourceManager::ResourceManager() {
    num_cores = std::thread::hardware_concurrency();

    for (int i = 0 ; i < get_num_cuda_devices() ; ++i) {
        DeviceID device_id = devices.size();
        auto device = new Device(device_id);

        devices.push_back(device);

        stream_devices.push_back(device_id);
        default_streams.push_back((StreamID)device_id);
    }
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

StreamID ResourceManager::create_stream(DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to create stream on non-existent device.");

    StreamID id = stream_devices.size();
    stream_devices.push_back(device_id);

    devices[id]->create_stream(id);

    return id;
}

EventID ResourceManager::create_event(DeviceID device_id) {
    if (device_id >= get_num_devices())
        ErrorManager::get_instance()->log_error(
            "Attempted to create stream on non-existent device.");

    EventID id = event_devices.size();
    event_devices.push_back(device_id);

    devices[id]->create_event(id);

    return id;
}

void ResourceManager::synchronize_event(EventID id) {
    if (id >= event_devices.size())
        ErrorManager::get_instance()->log_error(
            "Attempted to synchronize non-existent event.");
    devices[event_devices[id]]->synchronize_event(id);
}
