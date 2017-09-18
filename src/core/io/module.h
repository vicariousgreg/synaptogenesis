#ifndef module_h
#define module_h

#include <string>
#include <vector>

#include "util/property_config.h"
#include "io/buffer.h"

class Network;
class Layer;
class Module;
class ModuleConfig;
typedef Module* (*MODULE_BUILD_PTR)(Layer *layer, ModuleConfig *config);

class ModuleConfig : public PropertyConfig {
    public:
        ModuleConfig(std::string structure, std::string layer, std::string type) {
            this->set_property("type", type);
            this->set_property("structure", structure);
            this->set_property("layer", layer);
            this->set_property("params", "");
        }

        std::string get_type() { return get_property("type"); }
        std::string get_structure() { return get_property("structure"); }
        std::string get_layer() { return get_property("layer"); }

        /* Setter that returns self pointer */
        ModuleConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }
};

class Module {
    public:
        Module(Layer *layer) : layer(layer) { }
        virtual ~Module() {}

        /* Override to implement input and output functionality.
         * If unused, do not override */
        virtual void feed_input(Buffer *buffer) { }
        virtual void feed_expected(Buffer *buffer) { }
        virtual void report_output(Buffer *buffer, OutputType output_type) { }

        /* Override to indicate IO type
         * This is used by the environment to determine which hooks to call
         */
        virtual IOTypeMask get_type() = 0;

        Layer* const layer;

        // Get the IOType of a module subclass
        static IOTypeMask get_type(std::string module_type);
        static IOTypeMask get_type(ModuleConfig *config);

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

};


/* Macros for Module subclass Registry */
// Put this one in .cpp
#define REGISTER_MODULE(CLASS_NAME, STRING, TYPE) \
int CLASS_NAME::module_id = \
    Module::register_module(STRING, \
        TYPE, CLASS_NAME::build); \
IOTypeMask CLASS_NAME::io_type = TYPE; \
\
Module *CLASS_NAME::build(Layer *layer, ModuleConfig *config) { \
    return new CLASS_NAME(layer, config); \
}

// Put this one in .h at bottom of class definition
#define MODULE_MEMBERS \
    protected: \
        static Module *build(Layer *layer, ModuleConfig *config); \
        static int module_id; \
        static IOTypeMask io_type; \
        virtual IOTypeMask get_type() { return io_type; }


typedef std::vector<Module*> ModuleList;

#endif
