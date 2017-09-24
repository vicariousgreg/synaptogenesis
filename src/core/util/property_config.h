#ifndef property_config_h
#define property_config_h

#include <string>
#include <map>
#include <vector>

typedef std::pair<std::string, std::string> StringPair;
typedef std::vector<StringPair> StringPairList;

class PropertyConfig {
    public:
        PropertyConfig() { }
        PropertyConfig(StringPairList pairs);

        const StringPairList get_properties() const;
        bool has(std::string key) const;
        std::string remove_property(std::string key);
        std::string get(std::string key) const;
        std::string get(std::string key, std::string def_val) const;

    protected:
        void set_internal(std::string key, std::string value);

        std::map<std::string, std::string> properties;
        std::vector<std::string> keys;
};

#endif
