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
        return ARRAY{ 0, VOID_POINTER, nullptr };
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
        return ARRAY{ 0, VOID_POINTER, nullptr };
    }
}

ARRAY get_connection_data(STATE state, char* conn_name, char* key) {
    try {
        auto conn = ((State*)state)->network->get_connection(conn_name);
        auto ptr = ((State*)state)->get_connection_data(conn, key);
        return build_array(ptr);
    } catch (...) {
        return ARRAY{ 0, VOID_POINTER, nullptr };
    }
}

ARRAY get_weight_matrix(STATE state, char* conn_name) {
    try {
        auto conn = ((State*)state)->network->get_connection(conn_name);
        auto ptr = ((State*)state)->get_weight_matrix(conn);
        return build_array(ptr);
    } catch (...) {
        return ARRAY{ 0, VOID_POINTER, nullptr };
    }
}


bool run(NETWORK net, ENVIRONMENT env, STATE state, PROPS args) {
    try {
        Engine engine(
            new Context((Network*)net, (Environment*)env, (State*)state));

        auto context = engine.run(*((PropertyConfig*)args));
        auto state = context->get_state();
        delete context;
        return true;
    } catch(...) {
        return false;
    }
}


void destroy(void* obj) {
    delete obj;
}
