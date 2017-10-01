#ifndef extern_h
#define extern_h

typedef void* NETWORK;
typedef void* ENVIRONMENT;
typedef void* STATE;
typedef void* PROPS;

extern "C" PROPS create_properties();
extern "C" void add_property(PROPS properties, char* key, char* val);
extern "C" void add_child(PROPS properties, char* key, PROPS child);
extern "C" void add_to_array(PROPS properties, char* key, PROPS props);

extern "C" ENVIRONMENT create_environment(PROPS properties);
extern "C" ENVIRONMENT load_env(char* filename);
extern "C" bool save_env(ENVIRONMENT env, char* filename);

extern "C" NETWORK create_network(PROPS properties);
extern "C" NETWORK load_net(char* filename);
extern "C" bool save_net(NETWORK network, char* filename);
extern "C" STATE run(NETWORK net, ENVIRONMENT env, STATE state, PROPS args);

extern "C" void destroy(void* obj);

#endif
