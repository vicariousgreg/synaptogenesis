#include "extern.h"
#include "builder.h"
#include "context.h"
#include "network/network.h"
#include "state/state.h"
#include "io/environment.h"
#include "engine/engine.h"
#include "util/constants.h"
#include "util/callback_manager.h"

ARRAY build_array(BasePointer* ptr, bool owner) {
    POINTER_TYPE type = VOID_POINTER;
    if (ptr->get_type() == std::type_index(typeid(float)))
        type = FLOAT_POINTER;
    else if (ptr->get_type() == std::type_index(typeid(int)))
        type = INT_POINTER;

    return build_array(ptr->get(), ptr->get_size(), type, owner);
}

ARRAY build_array(void* ptr, int size, POINTER_TYPE type, bool owner) {
    ARRAY arr;
    arr.size = size;
    arr.type = type;
    arr.data = ptr;
    arr.owner = owner;
    return arr;
}

ARRAY null_array() {
    return ARRAY{ 0, VOID_POINTER, nullptr, false };
}

void free_array(ARRAY arr) {
    if (arr.owner) free(arr.data);
}

void free_array_deep(ARRAY arr) {
    if (arr.owner) {
        for (int i = 0 ; i < arr.size ; ++i)
            free(((void**)arr.data)[i]);
        free(arr.data);
    }
}

static ARRAY string_array(const std::vector<std::string>& strings) {
    char** pp = (char**)malloc(strings.size() * sizeof(char*));

    for (int i = 0 ; i < strings.size() ; ++i) {
        pp[i] = (char*)malloc((strings[i].length()+1) * sizeof(char));

        for (int j = 0 ; j < strings[i].length()+1 ; ++j)
            pp[i][j] = strings[i].c_str()[j];
    }

    return ARRAY{ strings.size(), STRING_POINTER, pp, true };
}

static ARRAY property_array(const ConfigArray& arr) {
    void** pp = (void**)malloc(arr.size() * sizeof(PropertyConfig*));
    for (int i = 0 ; i < arr.size() ; ++i) pp[i] = arr[i];
    return ARRAY{ arr.size(), PROPS_POINTER, pp, true };
}

PROPS create_properties() {
    return new PropertyConfig();
}

void add_property(PROPS properties, char* key, char* val) {
    ((PropertyConfig*)properties)->set(key, val);
}

void add_child(PROPS properties, char* key, PROPS child) {
    ((PropertyConfig*)properties)->set_child(key, (PropertyConfig*)child);
}

void add_to_array(PROPS properties, char* key, char* val) {
    ((PropertyConfig*)properties)->add_to_array(key, val);
}

void add_to_child_array(PROPS properties, char* key, PROPS props) {
    ((PropertyConfig*)properties)->add_to_child_array(key, (PropertyConfig*)props);
}

ARRAY get_keys(PROPS properties) {
    auto& keys = ((PropertyConfig*)properties)->get_keys();
    if (keys.size() > 0) return string_array(keys);
    else return null_array();
}

ARRAY get_child_keys(PROPS properties) {
    auto& keys = ((PropertyConfig*)properties)->get_child_keys();
    if (keys.size() > 0) return string_array(keys);
    else return null_array();
}

ARRAY get_array_keys(PROPS properties) {
    auto& keys = ((PropertyConfig*)properties)->get_array_keys();
    if (keys.size() > 0) return string_array(keys);
    else return null_array();
}

ARRAY get_child_array_keys(PROPS properties) {
    auto& keys = ((PropertyConfig*)properties)->get_child_array_keys();
    if (keys.size() > 0) return string_array(keys);
    else return null_array();
}


const char* get_property(PROPS properties, char* key) {
    if (((PropertyConfig*)properties)->has(key))
        return ((PropertyConfig*)properties)->get_c_str(key);
    else return nullptr;
}

PROPS get_child(PROPS properties, char* key) {
    if (((PropertyConfig*)properties)->has_child(key))
        return ((PropertyConfig*)properties)->get_child(key);
    else return nullptr;
}

ARRAY get_array(PROPS properties, char* key) {
    if (((PropertyConfig*)properties)->has_array(key)) {
        auto arr = ((PropertyConfig*)properties)->get_array(key);
        if (arr.size() > 0)
            return string_array(arr);
    }
    return null_array();
}

ARRAY get_child_array(PROPS properties, char* key) {
    if (((PropertyConfig*)properties)->has_child_array(key)) {
        auto arr = ((PropertyConfig*)properties)->get_child_array(key);
        if (arr.size() > 0)
            return property_array(arr);
    }
    return null_array();
}



ENVIRONMENT create_environment(PROPS properties) {
    return new Environment((PropertyConfig*)properties);
}

ENVIRONMENT load_env(char* filename) {
    try {
        return load_environment(filename);
    } catch (...) {
        return nullptr;
    }
}

bool save_env(ENVIRONMENT env, char* filename) {
    try {
        save_environment((Environment*)env, filename);
        return true;
    } catch (...) {
        return false;
    }
}


NETWORK create_network(PROPS properties) {
    return new Network(new NetworkConfig((PropertyConfig*)properties));
}

NETWORK load_net(char* filename) {
    try {
        return load_network(filename);
    } catch (...) {
        return nullptr;
    }
}

bool save_net(NETWORK network, char* filename) {
    try {
        save_network((Network*)network, filename);
        return true;
    } catch (...) {
        return false;
    }
}

STATE build_state(NETWORK net) {
    return new State((Network*)net);
}

bool load_state(STATE state, char* filename) {
    try {
        ((State*)state)->load(filename);
        return true;
    } catch (...) {
        return false;
    }
}

bool save_state(STATE state, char* filename) {
    try {
        ((State*)state)->save(filename);
        return true;
    } catch (...) {
        return false;
    }
}

ARRAY get_neuron_data(STATE state, char* structure_name,
        char* layer_name, char* key) {
    try {
        auto layer = ((State*)state)->network
            ->get_structure(structure_name)->get_layer(layer_name);
        auto ptr = ((State*)state)->get_neuron_data(layer, key);
        return build_array(ptr, false);
    } catch (...) {
        return null_array();
    }
}

ARRAY get_layer_data(STATE state, char* structure_name,
        char* layer_name, char* key) {
    try {
        auto layer = ((State*)state)->network
            ->get_structure(structure_name)->get_layer(layer_name);
        auto ptr = ((State*)state)->get_layer_data(layer, key);
        return build_array(ptr, false);
    } catch (...) {
        return null_array();
    }
}

ARRAY get_connection_data(STATE state, char* conn_name, char* key) {
    try {
        auto conn = ((State*)state)->network->get_connection(conn_name);
        auto ptr = ((State*)state)->get_connection_data(conn, key);
        return build_array(ptr, false);
    } catch (...) {
        return null_array();
    }
}

ARRAY get_weight_matrix(STATE state, char* conn_name, char* key) {
    if (key == nullptr) key = "weights";
    try {
        auto conn = ((State*)state)->network->get_connection(conn_name);
        auto ptr = ((State*)state)->get_weight_matrix(conn, key);
        return build_array(ptr, false);
    } catch (...) {
        return null_array();
    }
}


PROPS run(NETWORK net, ENVIRONMENT env, STATE state, PROPS args) {
    if (net == nullptr or state == nullptr) return nullptr;

// Do not catch exceptions in debug mode (so they can be traced in gdb)
#ifndef DEBUG
    try {
#endif
        return Engine(Context(
                        (Network*)net,
                        (Environment*)env,
                        (State*)state))
                      .run((args == nullptr)
                              ? PropertyConfig()
                              : *((PropertyConfig*)args));
#ifndef DEBUG
    } catch(...) {
        return nullptr;
    }
#endif
}


void destroy(void* obj) {
    delete obj;
}

int get_cpu() {
    return ResourceManager::get_instance()->get_host_id();
}

ARRAY get_gpus() {
    auto ids = ResourceManager::get_instance()->get_gpu_ids();
    int* p = (int*)malloc(ids.size() * sizeof(int));
    for (int i = 0 ; i < ids.size() ; ++i) p[i] = ids[i];
    return build_array(p, ids.size(), INT_POINTER, true);
}

ARRAY get_all_devices() {
    auto ids = ResourceManager::get_instance()->get_all_ids();
    int* p = (int*)malloc(ids.size() * sizeof(int));
    for (int i = 0 ; i < ids.size() ; ++i) p[i] = ids[i];
    return build_array(p, ids.size(), INT_POINTER, true);
}

void set_suppress_output(bool val) {
    Logger::suppress_output = val;
}

void set_warnings(bool val) {
    Logger::warnings = val;
}

void set_debug(bool val) {
    Logger::debug = val;
}

void interrupt_engine() {
    Engine::interrupt_async();
}

void add_io_callback(char* name, long long addr) {
    CallbackManager::get_instance()->add_io_callback(name,
        (void (*)(int, int, void*))(addr));
}

void add_distance_weight_callback(char* name, long long addr) {
    CallbackManager::get_instance()->add_distance_weight_callback(name,
        (void (*)(int, int, void*, void*))(addr));
}

void add_weight_callback(char* name, long long addr) {
    CallbackManager::get_instance()->add_weight_callback(name,
        (void (*)(int, int, void*))(addr));
}
