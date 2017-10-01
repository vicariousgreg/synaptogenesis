#include "extern.h"
#include "builder.h"
#include "network/network.h"
#include "io/environment.h"
#include "engine/engine.h"
#include "engine/context.h"
#include "util/constants.h"

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

STATE run(NETWORK net, ENVIRONMENT env, STATE state, PROPS args) {
    try {
        Engine engine(
            new Context((Network*)net, (Environment*)env, (State*)state));

        auto context = engine.run(*((PropertyConfig*)args));
        auto state = context->get_state();
        delete context;
        return state;
    } catch(...) {
        return nullptr;
    }
}


void destroy(void* obj) {
    delete obj;
}

