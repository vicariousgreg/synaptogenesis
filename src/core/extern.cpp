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

NETWORK create_network() {
    return new Network();
}

STRUCTURE create_structure(char* name, char* type) {
    return new Structure(name, ClusterTypes.at(type));
}

bool add_structure(NETWORK network, STRUCTURE structure) {
    try {
        ((Network*)network)->add_structure((Structure*)structure);
        return true;
    } catch (...) {
        return false;
    }
}

bool add_layer(STRUCTURE structure, PROPS props) {
    try {
        ((Structure*)structure)->add_layer(new LayerConfig((PropertyConfig*)props));
        return true;
    } catch (...) {
        return false;
    }
}

bool connect_layers(STRUCTURE from_structure, char* from_layer,
        STRUCTURE to_structure, char* to_layer, PROPS props) {
    try {
        std::string dendrite = ((PropertyConfig*)props)->get("dendrite", "root");
        Structure::connect(
            (Structure*)from_structure, from_layer,
            (Structure*)to_structure, to_layer,
            new ConnectionConfig((PropertyConfig*)props), dendrite);
        return true;
    } catch (...) {
        return false;
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

bool run(NETWORK network, int iterations, bool verbose) {
    Engine engine(new Context((Network*)network));
    delete engine.run(iterations, verbose);
}
