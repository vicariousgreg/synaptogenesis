#ifndef extern_h
#define extern_h

typedef void* NETWORK;
typedef void* STRUCTURE;
typedef void* PROPS;

extern "C" PROPS create_properties();
extern "C" void add_property(PROPS properties, char* key, char* val);
extern "C" void add_child(PROPS properties, char* key, PROPS child);
extern "C" void add_to_array(PROPS properties, char* key, PROPS props);

extern "C" NETWORK create_network(PROPS properties);

extern "C" bool save_net(NETWORK network, char* filename);
extern "C" bool run(NETWORK network, int iterations, bool verbose);

#endif
