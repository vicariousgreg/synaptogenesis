#ifndef property_config_h
#define property_config_h

#include <string>
#include <map>
#include <vector>

class PropertyConfig;

typedef std::pair<std::string, std::string> StringPair;
typedef std::vector<StringPair> StringPairList;

typedef std::pair<std::string, PropertyConfig*> PropertyPair;
typedef std::vector<PropertyPair> PropertyPairList;

class PropertyConfig {
    public:
        PropertyConfig() { }
        PropertyConfig(PropertyConfig *other);
        PropertyConfig(StringPairList pairs);

        virtual ~PropertyConfig();

        /* Get set of key-value pairs */
        const StringPairList get() const;
        const StringPairList get_alphabetical() const;

        /* Single property functions */
        bool has(std::string key) const;
        std::string get(std::string key) const;
        std::string get(std::string key, std::string def_val) const;
        PropertyConfig *set_value(std::string key, std::string val);
        std::string remove_property(std::string key);

        /* Get a set of key-child pairs */
        const PropertyPairList get_children() const;
        const PropertyPairList get_children_alphabetical() const;

        /* Single child functions */
        bool has_child(std::string key) const;
        PropertyConfig *get_child(std::string key) const;
        PropertyConfig *get_child(std::string key, PropertyConfig* def_val) const;
        void set_child(std::string key, PropertyConfig *child);
        PropertyConfig *remove_child(std::string key);

        std::string str() const;

    protected:
        std::map<std::string, std::string> properties;
        std::vector<std::string> keys;

        std::map<std::string, PropertyConfig*> children;
        std::vector<std::string> children_keys;
};

#endif
