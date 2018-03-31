#ifndef extern_h
#define extern_h

class BasePointer;

typedef enum {
    FLOAT_POINTER=1,
    INT_POINTER=2,
    STRING_POINTER=3,
    PROPS_POINTER=4,
    VOID_POINTER=5
} POINTER_TYPE;

typedef struct ARRAY {
    int size;
    POINTER_TYPE type;
    void* data;
    bool owner;
} ARRAY;

ARRAY build_array(BasePointer* ptr, bool owner);
ARRAY build_array(void* ptr, int size, POINTER_TYPE type, bool owner);
ARRAY null_array();
extern "C" void free_array(ARRAY arr);
extern "C" void free_array_deep(ARRAY arr);

typedef void* NETWORK;
typedef void* ENVIRONMENT;
typedef void* STATE;
typedef void* PROPS;

extern "C" PROPS create_properties();
extern "C" void add_property(PROPS properties, char* key, char* val);
extern "C" void add_child(PROPS properties, char* key, PROPS child);
extern "C" void add_to_array(PROPS properties, char* key, char* val);
extern "C" void add_to_child_array(PROPS properties, char* key, PROPS props);

extern "C" ARRAY get_keys(PROPS properties);
extern "C" ARRAY get_child_keys(PROPS properties);
extern "C" ARRAY get_array_keys(PROPS properties);
extern "C" ARRAY get_child_array_keys(PROPS properties);

/* Use the return value immediately, because it's unstable */
extern "C" const char* get_property(PROPS properties, char* key);
extern "C" PROPS get_child(PROPS properties, char* key);
extern "C" ARRAY get_array(PROPS properties, char* key);
extern "C" ARRAY get_child_array(PROPS properties, char* key);

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
extern "C" ARRAY get_layer_data(
    STATE state, char* structure_name, char* layer_name, char* key);
extern "C" ARRAY get_connection_data(STATE state, char* conn_name, char* key);
extern "C" ARRAY get_weight_matrix(STATE state, char* conn_name, char* key=nullptr);

extern "C" PROPS run(NETWORK net, ENVIRONMENT env, STATE state, PROPS args);

extern "C" void destroy(void* obj);

extern "C" int get_num_gpus();
extern "C" int get_cpu();
extern "C" ARRAY get_gpus();
extern "C" ARRAY get_all_devices();

extern "C" void set_suppress_output(bool val);
extern "C" void set_warnings(bool val);
extern "C" void set_debug(bool val);

extern "C" void interrupt_engine();

extern "C" void add_io_callback(char* name, long long addr);
extern "C" void add_weight_callback(char* name, long long addr);
extern "C" void add_distance_weight_callback(char* name, long long addr);

#endif
