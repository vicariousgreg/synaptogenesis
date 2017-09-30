#ifndef extern_h
#define extern_h

typedef void* NETWORK;
typedef void* STRUCTURE;
typedef void* PROPS;

extern "C" PROPS create_properties();
extern "C" void add_property(PROPS properties, char* key, char* val);
extern "C" void add_child(PROPS properties, char* key, PROPS child);

extern "C" NETWORK create_network();
extern "C" STRUCTURE create_structure(char* name, char* type);
extern "C" bool add_structure(NETWORK network, STRUCTURE structure);
extern "C" bool add_layer(STRUCTURE structure, PROPS props);
extern "C" bool connect_layers(STRUCTURE from_structure, char* from_layer,
    STRUCTURE to_structure, char* to_layer, PROPS props);

extern "C" bool save_net(NETWORK network, char* filename);
extern "C" bool run(NETWORK network, int iterations, bool verbose);

#endif
