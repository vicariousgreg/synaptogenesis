#include <algorithm>

#include "util/property_config.h"
#include "util/error_manager.h"

PropertyConfig::PropertyConfig(StringPairList pairs) {
    for (auto pair : pairs) {
        keys.push_back(pair.first);
        properties[pair.first] = pair.second;
    }
}

const StringPairList PropertyConfig::get() const {
    StringPairList pairs;
    for (auto key : keys)
        pairs.push_back(StringPair(key, properties.at(key)));
    return pairs;
}

bool PropertyConfig::has(std::string key) const
    { return properties.count(key) > 0; }

std::string PropertyConfig::remove_property(std::string key) {
    if (has(key)) {
        std::string value = get(key);
        properties.erase(key);
        keys.erase(std::find(keys.begin(), keys.end(), key));
        return value;
    }
    ErrorManager::get_instance()->log_error(
        "Attempted to remove non-existent property " + key
        + " from PropertyConfig!");
}

std::string PropertyConfig::get(std::string key) const
    { return properties.at(key); }

std::string PropertyConfig::get(std::string key, std::string def_val) const {
    if (has(key)) return properties.at(key);
    else                   return def_val;
}

void PropertyConfig::set_internal(std::string key, std::string value) {
    properties[key] = value;
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
        keys.push_back(key);
}

PropertyConfig* PropertyConfig::set_value(std::string key, std::string val) {
    set_internal(key, val);
    return this;
}

void PropertyConfig::set_child(std::string key, PropertyConfig *child) {
    children[key] = child;
    if (std::find(children_keys.begin(), children_keys.end(), key)
            == children_keys.end())
        children_keys.push_back(key);
}

bool PropertyConfig::has_child(std::string key)
    { return children.count(key) > 0; }

PropertyConfig* PropertyConfig::get_child(std::string key)
    { return children.at(key); }

const PropertyPairList PropertyConfig::get_children() {
    PropertyPairList pairs;
    for (auto key : children_keys)
        pairs.push_back(PropertyPair(key, children.at(key)));
    return pairs;
}
