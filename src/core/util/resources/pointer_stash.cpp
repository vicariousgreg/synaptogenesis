#include "util/resources/pointer_stash.h"

PointerStash *PointerStash::instance = nullptr;

PointerStash *PointerStash::get_instance() {
    if (PointerStash::instance == nullptr)
        PointerStash::instance = new PointerStash();
    return PointerStash::instance;
}

PointerStash::PointerStash() { }

PointerStash::~PointerStash() {
    this->clear();
}

void PointerStash::add(State* state, std::map<PointerKey, BasePointer*> new_pointers) {
    for (auto& pair : new_pointers) {
        owners[state][pair.first] = pair.second;
        pointers[pair.first] = pair.second;
    }
}

const std::map<PointerKey, BasePointer*> PointerStash::get(State* state) const {
    if (state == nullptr)
        return pointers;
    else
        try {
            return owners.at(state);
        } catch (std::out_of_range()) {
            return std::map<PointerKey, BasePointer*>();
        }
}

BasePointer* PointerStash::get(PointerKey key) const {
    try {
        return pointers.at(key);
    } catch (std::out_of_range) {
        return nullptr;
    }
}

void PointerStash::clear(State* state) {
    if (state == nullptr) {
        for (auto& pair : owners[state]) {
            delete pair.second;
            pointers.erase(pair.first);
        }
        owners.erase(state);
    } else {
        for (auto& pair : pointers)
            delete pair.second;
        pointers.clear();
        owners.clear();
    }
}
