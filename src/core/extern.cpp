#include "extern.h"
#include "builder.h"
#include "network/network.h"
#include "state/state.h"
#include "io/environment.h"
#include "engine/engine.h"
#include "engine/context.h"
#include "util/constants.h"

ARRAY build_array(BasePointer* ptr) {
    ARRAY arr;
    arr.size = ptr->get_size();
    arr.data = ptr->get();
    if (ptr->get_type() == std::type_index(typeid(float)))
        arr.type = FLOAT_POINTER;
    else if (ptr->get_type() == std::type_index(typeid(int)))
        arr.type = INT_POINTER;
    else if (ptr->get_type() == std::type_index(typeid(void)))
        arr.type = VOID_POINTER;
    return arr;
}

ARRAY null_array() {
    return ARRAY{ 0, VOID_POINTER, nullptr };
}

void free_array(ARRAY arr) {
    free(arr.data);
}

void free_array_deep(ARRAY arr) {
    for (int i = 0 ; i < arr.size ; ++i)
        free(((void**)arr.data)[i]);
    free(arr.data);
}

static ARRAY string_array(const std::vector<std::string>& strings) {
    char** pp = (char**)malloc(strings.size() * sizeof(char*));

    for (int i = 0 ; i < strings.size() ; ++i) {
        pp[i] = (char*)malloc((strings[i].length()+1) * sizeof(char));

        for (int j = 0 ; j < strings[i].length()+1 ; ++j)
            pp[i][j] = strings[i].c_str()[j];
    }

    return ARRAY{ strings.size(), STRING_POINTER, pp };
}

static ARRAY property_array(const ConfigArray& arr) {
    void** pp = (void**)malloc(arr.size() * sizeof(PropertyConfig*));
    for (int i = 0 ; i < arr.size() ; ++i) pp[i] = arr[i];
    return ARRAY{ arr.size(), PROPS_POINTER, pp };
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

void add_to_array(PROPS properties, char* key, PROPS props) {
    ((PropertyConfig*)properties)->add_to_array(key, (PropertyConfig*)props);
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
        return build_array(ptr);
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
        return build_array(ptr);
    } catch (...) {
        return null_array();
    }
}

ARRAY get_connection_data(STATE state, char* conn_name, char* key) {
    try {
        auto conn = ((State*)state)->network->get_connection(conn_name);
        auto ptr = ((State*)state)->get_connection_data(conn, key);
        return build_array(ptr);
    } catch (...) {
        return null_array();
    }
}

ARRAY get_weight_matrix(STATE state, char* conn_name) {
    try {
        auto conn = ((State*)state)->network->get_connection(conn_name);
        auto ptr = ((State*)state)->get_weight_matrix(conn);
        return build_array(ptr);
    } catch (...) {
        return null_array();
    }
}


PROPS run(NETWORK net, ENVIRONMENT env, STATE state, PROPS args) {
    try {
        if (net == nullptr or state == nullptr) return nullptr;

        return Engine(Context(
                        (Network*)net,
                        (Environment*)env,
                        (State*)state))
                      .run((args == nullptr)
                              ? PropertyConfig()
                              : *((PropertyConfig*)args));
    } catch(...) {
        return nullptr;
    }
}


void destroy(void* obj) {
    delete obj;
}
