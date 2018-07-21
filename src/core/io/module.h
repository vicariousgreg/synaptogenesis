#ifndef module_h
#define module_h

#include <string>
#include <vector>

#include "state/attributes.h"
#include "io/buffer.h"
#include "util/property_config.h"
#include "report.h"

class Network;
class Layer;
class Module;
class ModuleConfig;
typedef std::vector<Layer*> LayerList;
typedef Module* (*MODULE_BUILD_PTR)(LayerList layers, ModuleConfig *config);

class ModuleConfig : public PropertyConfig {
    public:
        ModuleConfig(PropertyConfig *config);
        ModuleConfig(std::string type);
        ModuleConfig(std::string type, std::string structure, std::string layer);

        ModuleConfig* add_layer(std::string structure, std::string layer);
        ModuleConfig* add_layer(PropertyConfig *config);

        std::string get_type() const { return get("type", ""); }
        const ConfigArray get_layers() const
            { return get_child_array("layers"); }
        const PropertyConfig* get_layer(Layer *layer) const;

    protected:
        std::map<std::string,
            std::map<std::string, PropertyConfig*>> layer_map;
};

class Module {
    public:
        virtual ~Module() { delete config; }

        static Module* build_module(Network *network, ModuleConfig *config);

        /* Module API
         * These functions call subclass implementations (see below) */
        void feed_input(Buffer *buffer);
        void report_output(Buffer *buffer);
        void cycle();

        /* Adds properties to an engine report */
        virtual void report(Report* report) { }

        /* Used by the environment to determine which hooks to call */
        IOTypeMask get_io_type(Layer *layer) const;
        KeySet get_input_keys(Layer *layer) const;
        KeySet get_output_keys(Layer *layer) const;

        /* Get the expected number of iterations according to a module
         * If the module is agnostic, it will return 0 */
        virtual size_t get_expected_iterations() const { return 0; }

        /* Gets the name of a module */
        virtual std::string get_name() const = 0;

        /* Checks if two modules are active simultaneously */
        bool is_coactive(Module* other) const;

        const LayerList layers;
        ModuleConfig* const config;

    protected:
        Module(LayerList layers, ModuleConfig *config);

        /* Override to implement IO functionality and module state cycling.
         * If unused, do not override */
        virtual void feed_input_impl(Buffer *buffer) { }
        virtual void report_output_impl(Buffer *buffer) { }
        virtual void cycle_impl() { }

        class ModuleBank {
            public:
                // Set of module implementations
                std::set<std::string> modules;
                std::map<std::string, MODULE_BUILD_PTR> build_pointers;
        };

        static int register_module(std::string module_type,
            MODULE_BUILD_PTR build_ptr);
        static ModuleBank* get_module_bank();

        void enforce_specified_io_type(std::string type);
        void enforce_unique_io_type(std::string type);
        void enforce_single_io_type(std::string type);
        void enforce_single_layer(std::string type);
        void enforce_equal_layer_sizes(std::string type);
        void set_default_io_type(IOTypeMask io_type);
        void set_io_type(IOTypeMask io_type);
        void set_io_type(Layer *layer, IOTypeMask io_type);
        void add_input_auxiliary_key(std::string key);
        void add_input_auxiliary_key(Layer *layer, std::string key);
        void add_output_auxiliary_key(std::string key);
        void add_output_auxiliary_key(Layer *layer, std::string key);
        void add_missing_keys();
        OutputType get_output_type(Layer* layer);

        std::map<Layer*, OutputType> output_types;
        std::map<Layer*, IOTypeMask> io_types;
        LayerKeyMap input_keys;
        LayerKeyMap output_keys;
        bool verbose;
        int start_delay;
        int cutoff;
        int curr_iteration;
        int rate;
};


/* Macros for Module subclass Registry */
// Put this one in .cpp
#define REGISTER_MODULE(CLASS_NAME, STRING) \
int CLASS_NAME::module_id = \
    Module::register_module(STRING, CLASS_NAME::build); \
\
Module *CLASS_NAME::build(LayerList layers, ModuleConfig *config) { \
    return new CLASS_NAME(layers, config); \
} \
std::string CLASS_NAME::get_name() const { return STRING; }

// Put this one in .h at bottom of class definition
#define MODULE_MEMBERS \
    protected: \
        static Module *build(LayerList layers, ModuleConfig *config); \
        static int module_id; \
        virtual std::string get_name() const;


typedef std::vector<Module*> ModuleList;

#endif
