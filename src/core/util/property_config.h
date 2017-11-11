#ifndef property_config_h
#define property_config_h

#include <string>
#include <map>
#include <vector>
#include <initializer_list>

class PropertyConfig;

typedef std::vector<PropertyConfig*> ConfigArray;
typedef std::vector<std::string> StringArray;

typedef std::pair<std::string, std::string> StringPair;
typedef std::vector<StringPair> StringPairList;

typedef std::pair<std::string, PropertyConfig*> PropertyPair;
typedef std::vector<PropertyPair> PropertyPairList;

typedef std::pair<std::string, ConfigArray> ConfigArrayPair;
typedef std::vector<ConfigArrayPair> ConfigArrayPairList;

typedef std::pair<std::string, StringArray> StringArrayPair;
typedef std::vector<StringArrayPair> StringArrayPairList;

class PropertyConfig {
    public:
        PropertyConfig() { }
        PropertyConfig(const PropertyConfig *other);
        PropertyConfig(std::initializer_list<StringPair> pairs);

        virtual ~PropertyConfig();

        /* Get set of key-value pairs */
        const StringPairList get() const;
        const StringPairList get_alphabetical() const;
        const StringArray get_keys() const { return keys; }

        /* Single property functions */
        bool has(std::string key) const;
        std::string get(std::string key, std::string def_val="") const;
        const char* get_c_str(std::string key) const;
        PropertyConfig *set(std::string key, std::string val);
        PropertyConfig *set(std::string key, char* val);
        PropertyConfig *set(std::string key, int val);
        PropertyConfig *set(std::string key, float val);
        PropertyConfig *set(std::string key, double val);
        PropertyConfig *set(std::string key, bool val);
        std::string remove_property(std::string key);

        /* Special typed getters */
        int get_int(std::string key, int def_val) const;
        float get_float(std::string key, float def_val) const;
        bool get_bool(std::string key, bool def_val) const;

        /* Get a set of key-child pairs */
        const PropertyPairList get_children() const;
        const PropertyPairList get_children_alphabetical() const;
        const StringArray get_child_keys() const
            { return children_keys; }

        /* Single child functions */
        bool has_child(std::string key) const;
        PropertyConfig *get_child(
            std::string key, PropertyConfig* def_val=nullptr) const;
        PropertyConfig *set_child(std::string key, const PropertyConfig *child);
        PropertyConfig *remove_child(std::string key);

        /* Get a set of key-array pairs */
        const StringArrayPairList get_arrays() const;
        const StringArrayPairList get_arrays_alphabetical() const;
        const StringArray get_array_keys() const { return string_array_keys; }

        /* Single array functions */
        bool has_array(std::string key) const;
        const StringArray get_array(std::string key) const;
        PropertyConfig *set_array(std::string key, StringArray array);
        PropertyConfig *add_to_array(std::string key, std::string val);
        std::string remove_from_array(std::string key, std::string val);
        std::string remove_from_array(std::string key, int index);
        StringArray remove_array(std::string key);

        /* Get a set of key - child array pairs */
        const ConfigArrayPairList get_child_arrays() const;
        const ConfigArrayPairList get_child_arrays_alphabetical() const;
        const StringArray get_child_array_keys() const
            { return child_array_keys; }

        /* Single child array functions */
        bool has_child_array(std::string key) const;
        const ConfigArray get_child_array(std::string key) const;
        PropertyConfig *set_child_array(std::string key, ConfigArray array);
        PropertyConfig *add_to_child_array(
            std::string key, const PropertyConfig* config);
        PropertyConfig *remove_from_child_array(
            std::string key, const PropertyConfig* config);
        PropertyConfig *remove_from_child_array(std::string key, int index);
        ConfigArray remove_child_array(std::string key);

        std::string str() const;

    protected:
        std::map<std::string, std::string> properties;
        StringArray keys;

        std::map<std::string, PropertyConfig*> children;
        StringArray children_keys;

        std::map<std::string, StringArray> string_arrays;
        StringArray string_array_keys;

        std::map<std::string, ConfigArray> child_arrays;
        StringArray child_array_keys;
};

#endif
