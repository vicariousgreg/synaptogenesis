#include "util/callback_manager.h"
#include "util/error_manager.h"

CallbackManager *CallbackManager::instance = nullptr;

CallbackManager *CallbackManager::get_instance() {
    if (CallbackManager::instance == nullptr)
        CallbackManager::instance = new CallbackManager();
    return CallbackManager::instance;
}

void CallbackManager::add_io_callback(std::string name,
        void (*addr)(int, int, void*)) {
    try {
        io_callbacks.at(name);
        LOG_ERROR("Duplicate IO callback: " + name);
    } catch (std::out_of_range) {
        io_callbacks[name] = addr;
    }
}

void CallbackManager::add_distance_weight_callback(std::string name,
        void (*addr)(int, int, void*, void*)) {
    try {
        distance_weight_callbacks.at(name);
        LOG_ERROR("Duplicate distance weight callback: " + name);
    } catch (std::out_of_range) {
        distance_weight_callbacks[name] = addr;
    }
}

void CallbackManager::add_weight_callback(std::string name,
        void (*addr)(int, int, void*)) {
    try {
        weight_callbacks.at(name);
        LOG_ERROR("Duplicate weight callback: " + name);
    } catch (std::out_of_range) {
        weight_callbacks[name] = addr;
    }
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

void (*CallbackManager::get_distance_weight_callback(std::string name))
        (int, int, void*, void*) {
    try {
        return distance_weight_callbacks.at(name);
    } catch (std::out_of_range) {
        LOG_ERROR("Could not find distance weight callback: " + name);
    }
}
