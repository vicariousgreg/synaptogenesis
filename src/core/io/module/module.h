#ifndef module_h
#define module_h

#include <string>
#include <vector>

#include "model/layer.h"
#include "io/buffer.h"

class Module;
typedef Module* (*MODULE_BUILD_PTR)(Layer *layer, std::string params);

class ModuleConfig {
    public:
        ModuleConfig(std::string name, std::string params)
            : name(name), params(params) { }

    const std::string name;
    const std::string params;
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
        static IOTypeMask get_module_type(std::string module_name);

        static Module* build_module(Layer *layer, ModuleConfig *config);

    protected:
        class ModuleBank {
            public:
                // Set of module implementations
                std::set<std::string> modules;
                std::map<std::string, MODULE_BUILD_PTR> build_pointers;
                std::map<std::string, IOTypeMask> types;
        };

        static int register_module(std::string module_name,
            IOTypeMask type, MODULE_BUILD_PTR build_ptr);
        static ModuleBank* get_module_bank();

};


/* Macros for Module subclass Registry */
// Put this one in .cpp
#define REGISTER_MODULE(CLASS_NAME, STRING, TYPE) \
int CLASS_NAME::module_id = \
    Module::register_module(STRING, \
        TYPE, CLASS_NAME::build); \
\
Module *CLASS_NAME::build(Layer *layer, std::string params) { \
    return new CLASS_NAME(layer, params); \
}

// Put this one in .h at bottom of class definition
#define MODULE_MEMBERS \
    protected: \
        static Module *build(Layer *layer, std::string params); \
        static int module_id;

typedef std::vector<Module*> ModuleList;

#endif
