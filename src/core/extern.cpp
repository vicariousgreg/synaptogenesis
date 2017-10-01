#include "extern.h"
#include "builder.h"
#include "network/network.h"
#include "network/structure.h"
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


NETWORK create_network(PROPS properties) {
    return new Network(new NetworkConfig((PropertyConfig*)properties));
}

bool save_net(NETWORK network, char* filename) {
    try {
        save_network((Network*)network, filename);
        return true;
    } catch (...) {
        return false;
    }
}

bool run(NETWORK network, int iterations, bool verbose) {
    Engine engine(new Context((Network*)network));
    delete engine.run(iterations, verbose);
}


void destroy(void* obj) {
    delete obj;
}

