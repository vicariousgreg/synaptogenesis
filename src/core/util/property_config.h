#ifndef property_config_h
#define property_config_h

#include <string>
#include <map>

class PropertyConfig {
    public:
        PropertyConfig() { }

        PropertyConfig(std::map<std::string, std::string> props)
            : properties(props) { }

        const std::map<std::string, std::string> get_properties() const
            { return properties; }

        bool has_property(std::string key) const
            { return properties.count(key) > 0; }

        std::string remove_property(std::string key) {
            std::string value = get_property(key);
            properties.erase(key);
            return value;
        }

        std::string get_property(std::string key) const
            { return properties.at(key); }

        std::string get_property(std::string key, std::string def_val) const {
            if (has_property(key)) return properties.at(key);
            else                   return def_val;
        }

    protected:
        void set_property_internal(std::string key, std::string value)
            { properties[key] = value; }

        std::map<std::string, std::string> properties;
};

#endif
