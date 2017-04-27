#ifndef property_config_h
#define property_config_h

#include <string>
#include <map>

class PropertyConfig {
    public:
        const std::map<std::string, std::string> get_properties() const
            { return properties; }
        std::string get_property(std::string key) const
            { return properties.at(key); }
        void set_property_internal(std::string key, std::string value)
            { properties[key] = value; }

    protected:
        std::map<std::string, std::string> properties;
};

#endif
