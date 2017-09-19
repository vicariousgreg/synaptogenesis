#ifndef module_h
#define module_h

#include <string>
#include <vector>

#include "state/attributes.h"
#include "util/property_config.h"
#include "io/buffer.h"

class Network;
class Layer;
class Module;
class ModuleConfig;
typedef std::vector<Layer*> LayerList;
typedef Module* (*MODULE_BUILD_PTR)(LayerList layers, ModuleConfig *config);

class ModuleConfig : public PropertyConfig {
    public:
        ModuleConfig(std::string type);
        ModuleConfig(std::string type, std::string structure, std::string layer);

        ModuleConfig* add_layer(std::string structure, std::string layer);
        ModuleConfig* add_layer(PropertyConfig *config);

        std::string get_type() const { return get_property("type"); }
        const std::vector<PropertyConfig*> get_layers() const { return layers; }

        /* Setter that returns self pointer */
        ModuleConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }

    protected:
        std::vector<PropertyConfig*> layers;
};

class Module {
    public:
        Module(LayerList layers);
        virtual ~Module() {}

        /* Override to implement IO functionality and module state cycling.
         * If unused, do not override */
        virtual void feed_input(Buffer *buffer) { }
        virtual void feed_expected(Buffer *buffer) { }
        virtual void report_output(Buffer *buffer) { }
        virtual void cycle() { };

        /* Override to indicate IO type
         * This is used by the environment to determine which hooks to call
         */
        virtual IOTypeMask get_io_type(Layer *layer) = 0;

        const LayerList layers;

        static Module* build_module(Network *network, ModuleConfig *config);

    protected:
        class ModuleBank {
            public:
                // Set of module implementations
                std::set<std::string> modules;
                std::map<std::string, MODULE_BUILD_PTR> build_pointers;
                std::map<std::string, IOTypeMask> io_types;
        };

        static int register_module(std::string module_type,
            IOTypeMask type, MODULE_BUILD_PTR build_ptr);
        static ModuleBank* get_module_bank();

        void enforce_single_layer(std::string type);
        void enforce_equal_layer_sizes(std::string type);

        std::map<Layer*, OutputType> output_types;
};


/* Macros for Module subclass Registry */
// Put this one in .cpp
#define REGISTER_MODULE(CLASS_NAME, STRING, TYPE) \
int CLASS_NAME::module_id = \
    Module::register_module(STRING, \
        TYPE, CLASS_NAME::build); \
IOTypeMask CLASS_NAME::io_type = TYPE; \
\
Module *CLASS_NAME::build(LayerList layers, ModuleConfig *config) { \
    return new CLASS_NAME(layers, config); \
}

// Put this one in .h at bottom of class definition
#define MODULE_MEMBERS \
    protected: \
        static Module *build(LayerList layers, ModuleConfig *config); \
        static int module_id; \
        static IOTypeMask io_type; \
        virtual IOTypeMask get_io_type(Layer *layer) { return io_type; }


typedef std::vector<Module*> ModuleList;

#endif
