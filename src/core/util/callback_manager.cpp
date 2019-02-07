#include "util/callback_manager.h"
#include "util/logger.h"

CallbackManager *CallbackManager::instance = nullptr;

CallbackManager *CallbackManager::get_instance() {
    if (CallbackManager::instance == nullptr)
        CallbackManager::instance = new CallbackManager();
    return CallbackManager::instance;
}

void CallbackManager::add_io_callback(std::string name,
        void (*addr)(int, int, void*)) {
    io_callbacks[name] = addr;
}

void CallbackManager::add_weight_callback(std::string name,
        void (*addr)(int, int, void*)) {
    weight_callbacks[name] = addr;
}

void CallbackManager::add_indices_weight_callback(std::string name,
        void (*addr)(int, int, void*, void*, void*, void*)) {
    indices_weight_callbacks[name] = addr;
}

void CallbackManager::add_distance_weight_callback(std::string name,
        void (*addr)(int, int, void*, void*)) {
    distance_weight_callbacks[name] = addr;
}

void CallbackManager::add_delay_weight_callback(std::string name,
        void (*addr)(int, int, void*, void*)) {
    delay_weight_callbacks[name] = addr;
}

void (*CallbackManager::get_io_callback(std::string name))(int, int, void*) {
    try {
        return io_callbacks.at(name);
    } catch (std::out_of_range) {
        LOG_ERROR("Could not find IO callback: " + name);
    }
}

void (*CallbackManager::get_weight_callback(std::string name))
        (int, int, void*) {
    try {
        return weight_callbacks.at(name);
    } catch (std::out_of_range) {
        LOG_ERROR("Could not find weight callback: " + name);
    }
}

void (*CallbackManager::get_indices_weight_callback(std::string name))
        (int, int, void*, void*, void*, void*) {
    try {
        return indices_weight_callbacks.at(name);
    } catch (std::out_of_range) {
        LOG_ERROR("Could not find indices weight callback: " + name);
    }
}

void (*CallbackManager::get_distance_weight_callback(std::string name))
        (int, int, void*, void*) {
    try {
        return distance_weight_callbacks.at(name);
    } catch (std::out_of_range) {
        LOG_ERROR("Could not find distance weight callback: " + name);
    }
}


void (*CallbackManager::get_delay_weight_callback(std::string name))
        (int, int, void*, void*) {
    try {
        return delay_weight_callbacks.at(name);
    } catch (std::out_of_range) {
        LOG_ERROR("Could not find delay weight callback: " + name);
    }
}
