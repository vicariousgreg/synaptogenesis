#ifndef extern_h
#define extern_h

typedef struct ARRAY {
    int size;
    float* data;
} ARRAY;

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

extern "C" STATE build_state(NETWORK net);
extern "C" bool load_state(STATE state, char* filename);
extern "C" bool save_state(STATE state, char* filename);

extern "C" ARRAY get_neuron_data(
    STATE state, char* structure_name, char* layer_name, char* key);
extern "C" float get_layer_data(
    STATE state, char* structure_name, char* layer_name, char* key);
extern "C" float get_connection_data(STATE state, char* conn_name, char* key);
extern "C" ARRAY get_weight_matrix(STATE state, char* conn_name);

extern "C" bool run(NETWORK net, ENVIRONMENT env, STATE state, PROPS args);

extern "C" void destroy(void* obj);

#endif
