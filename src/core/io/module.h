#ifndef module_h
#define module_h

#include <string>
#include <vector>

#include "state/attributes.h"
#include "engine/report.h"
#include "io/buffer.h"
#include "util/property_config.h"

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

        ModuleConfig* add_layer(std::string structure, std::string layer,
            std::string params="");
        ModuleConfig* add_layer(PropertyConfig *config);

        std::string get_type() const { return get("type", ""); }
        const ConfigArray get_layers() const
            { return get_array("layers"); }
        const PropertyConfig* get_layer(Layer *layer) const;

        /* Setters that returns self pointer */
        ModuleConfig *set(std::string key, std::string value) {
            PropertyConfig::set(key, value);
            return this;
        }
        ModuleConfig *set_child(std::string key, PropertyConfig* child) {
            PropertyConfig::set_child(key, child);
            return this;
        }
        ModuleConfig *set_array(std::string key, ConfigArray array) {
            PropertyConfig::set_array(key, array);
            return this;
        }
        ModuleConfig *add_to_array(std::string key, PropertyConfig* config) {
            PropertyConfig::add_to_array(key, config);
            return this;
        }

    protected:
        std::map<std::string,
            std::map<std::string, PropertyConfig*>> layer_map;
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

        /* Adds properties to an engine report */
        virtual void report(Report* report) { }

        /* Override to indicate IO type
         * This is used by the environment to determine which hooks to call
         */
        IOTypeMask get_io_type(Layer *layer) { return io_types.at(layer); }

        /* Gets the name of a module */
        virtual std::string get_name() = 0;

        const LayerList layers;

        static Module* build_module(Network *network, ModuleConfig *config);

    protected:
        class ModuleBank {
            public:
                // Set of module implementations
                std::set<std::string> modules;
                std::map<std::string, MODULE_BUILD_PTR> build_pointers;
        };

        static int register_module(std::string module_type,
            MODULE_BUILD_PTR build_ptr);
        static ModuleBank* get_module_bank();

        void enforce_single_layer(std::string type);
        void enforce_equal_layer_sizes(std::string type);
        void set_io_type(IOTypeMask io_type);
        void set_io_type(Layer *layer, IOTypeMask io_type);
        OutputType get_output_type(Layer* layer);

        std::map<Layer*, OutputType> output_types;
        std::map<Layer*, IOTypeMask> io_types;
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
std::string CLASS_NAME::get_name() { return STRING; }

// Put this one in .h at bottom of class definition
#define MODULE_MEMBERS \
    protected: \
        static Module *build(LayerList layers, ModuleConfig *config); \
        static int module_id; \
        virtual std::string get_name();


typedef std::vector<Module*> ModuleList;

#endif
