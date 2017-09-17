#ifndef environment_model_h
#define environment_model_h

#include <map>
#include <vector>

#include "io/module/module.h"

class EnvironmentModel {
    public:
        EnvironmentModel() { }
        virtual ~EnvironmentModel() { remove_modules(); }

        void add_module(ModuleConfig* config);
        void remove_modules();
        void remove_modules(std::string structure, std::string layer="");

        const std::vector<ModuleConfig*>& get_modules() { return config_list; }
        IOTypeMask get_type(std::string structure, std::string layer);

    private:
        std::map<std::string,
            std::map<std::string,
                std::vector<ModuleConfig*> > > config_map;
        std::map<std::string,
            std::map<std::string, IOTypeMask > > io_type_map;
        std::vector<ModuleConfig*> config_list;
};

#endif
