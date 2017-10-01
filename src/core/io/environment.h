#ifndef environment_h
#define environment_h

#include "util/property_config.h"

class Environment : public PropertyConfig {
    public:
        Environment() { }
        Environment(PropertyConfig *config) : PropertyConfig(config) { }

        /* Save or load environment to/from JSON file */
        static Environment* load(std::string path);
        void save(std::string path);

        /* Add or remove modules */
        void add_module(PropertyConfig* config);
        void remove_modules();
        void remove_modules(std::string structure,
            std::string layer="", std::string type="");

        /* Getters */
        const ConfigArray get_modules() { return get_array("modules") ; }
};

#endif
