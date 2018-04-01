#include <algorithm>

#include "util/property_config.h"
#include "util/logger.h"

PropertyConfig::PropertyConfig(const PropertyConfig *other) {
    for (auto pair : other->get())
        this->set(pair.first, pair.second);
    for (auto pair : other->get_children())
        this->set_child(pair.first, pair.second);
    for (auto pair : other->get_arrays()) {
        for (auto str : pair.second)
            this->add_to_array(pair.first, str);
    }
    for (auto pair : other->get_child_arrays()) {
        for (auto config : pair.second)
            this->add_to_child_array(pair.first, config);
    }
}

PropertyConfig::PropertyConfig(std::initializer_list<StringPair> pairs) {
    for (auto pair : pairs) {
        keys.push_back(pair.first);
        properties[pair.first] = pair.second;
    }
}

PropertyConfig::~PropertyConfig() {
    for (auto pair : children) delete pair.second;
    for (auto pair : child_arrays)
        for (auto config : pair.second)
            delete config;
}


const StringPairList PropertyConfig::get() const {
    StringPairList pairs;
    for (auto key : keys)
        pairs.push_back(StringPair(key, properties.at(key)));
    return pairs;
}

const StringPairList PropertyConfig::get_alphabetical() const {
    StringPairList pairs;
    for (auto pair : properties)
        pairs.push_back(StringPair(pair.first, pair.second));
    return pairs;
}


bool PropertyConfig::has(std::string key) const
    { return properties.count(key) > 0; }

std::string PropertyConfig::get(std::string key, std::string def_val) const {
    if (has(key)) return properties.at(key);
    else          return def_val;
}

const char* PropertyConfig::get_c_str(std::string key) const {
    if (has(key)) return properties.at(key).c_str();
    else          return nullptr;
}

PropertyConfig* PropertyConfig::set(std::string key, std::string value) {
    properties[key] = value;
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
        keys.push_back(key);
    return this;
}

PropertyConfig* PropertyConfig::set(std::string key, char* value)
    { return set(key, std::string(value)); }
PropertyConfig* PropertyConfig::set(std::string key, int value)
    { return set(key, std::to_string(value)); }
PropertyConfig* PropertyConfig::set(std::string key, float value)
    { return set(key, std::to_string(value)); }
PropertyConfig* PropertyConfig::set(std::string key, double value)
    { return set(key, std::to_string(value)); }
PropertyConfig* PropertyConfig::set(std::string key, bool value)
    { return set(key, ((value) ? std::string("true") : std::string("false"))); }

std::string PropertyConfig::remove_property(std::string key) {
    if (has(key)) {
        std::string value = get(key);
        properties.erase(key);
        keys.erase(std::find(keys.begin(), keys.end(), key));
        return value;
    }
    LOG_ERROR(
        "Attempted to remove non-existent property " + key
        + " from PropertyConfig!");
}

int PropertyConfig::get_int(std::string key, int def_val) const {
    try {
        if (has(key)) return std::stoi(get(key));
        else          return def_val;
    } catch (std::invalid_argument) {
        LOG_ERROR(
            "Could not convert property \""
            + key + "\" (" + get(key) + ") to integer!");
    }
}

float PropertyConfig::get_float(std::string key, float def_val) const {
    try {
        if (has(key)) return std::stof(get(key));
        else          return def_val;
    } catch (std::invalid_argument) {
        LOG_ERROR(
            "Could not convert property \""
            + key + "\" (" + get(key) + ") to float!");
    }
}

bool PropertyConfig::get_bool(std::string key, bool def_val) const {
    if (has(key)) return get(key) == "true";
    else          return def_val;
}



const PropertyPairList PropertyConfig::get_children() const {
    PropertyPairList pairs;
    for (auto key : children_keys)
        pairs.push_back(PropertyPair(key, children.at(key)));
    return pairs;
}

const PropertyPairList PropertyConfig::get_children_alphabetical() const {
    PropertyPairList pairs;
    for (auto pair : children)
        pairs.push_back(PropertyPair(pair.first, pair.second));
    return pairs;
}


bool PropertyConfig::has_child(std::string key) const
    { return children.count(key) > 0; }

PropertyConfig* PropertyConfig::get_child(
        std::string key, PropertyConfig *def_val) const {
    if (has_child(key)) return children.at(key);
    else                return def_val;
}

PropertyConfig *PropertyConfig::set_child(std::string key, const PropertyConfig *child) {
    children[key] = new PropertyConfig(child);
    if (std::find(children_keys.begin(), children_keys.end(), key)
            == children_keys.end())
        children_keys.push_back(key);
    return this;
}

PropertyConfig* PropertyConfig::remove_child(std::string key) {
    if (has_child(key)) {
        PropertyConfig* child = get_child(key);
        children.erase(key);
        children_keys.erase(
            std::find(children_keys.begin(), children_keys.end(), key));
        return child;
    }
    LOG_ERROR(
        "Attempted to remove non-existent property " + key
        + " from PropertyConfig!");
}


const StringArrayPairList PropertyConfig::get_arrays() const {
    StringArrayPairList pairs;
    for (auto key : string_array_keys)
        pairs.push_back(StringArrayPair(key, string_arrays.at(key)));
    return pairs;
}

const StringArrayPairList PropertyConfig::get_arrays_alphabetical() const {
    StringArrayPairList pairs;
    for (auto pair : string_arrays)
        pairs.push_back(StringArrayPair(pair.first, pair.second));
    return pairs;
}


bool PropertyConfig::has_array(std::string key) const
    { return string_arrays.count(key) > 0; }

const StringArray PropertyConfig::get_array(std::string key) const {
    if (has_array(key)) return string_arrays.at(key);
    else                return StringArray();
}

PropertyConfig *PropertyConfig::set_array(std::string key, StringArray array) {
    string_arrays[key] = array;
    if (std::find(string_array_keys.begin(), string_array_keys.end(), key)
            == string_array_keys.end())
        string_array_keys.push_back(key);
    return this;
}

PropertyConfig *PropertyConfig::add_to_array(std::string key, std::string val) {
    if (not has_array(key)) set_array(key, StringArray());
    string_arrays.at(key).push_back(val);
    return this;
}

std::string PropertyConfig::remove_from_array(
        std::string key, std::string val) {
    if (has_array(key)) {
        auto arr = string_arrays.at(key);
        auto it = std::find(arr.begin(), arr.end(), val);
        if (it != arr.end()) {
            auto to_return = *it;
            arr.erase(it);
            return to_return;
        } else
            LOG_ERROR(
                "Attempted to remove non-existent property " + key
                + " from PropertyConfig array " + key + "!");
    } else {
        LOG_ERROR(
            "Attempted to remove non-existent property " + key
            + " from PropertyConfig array " + key + "!");
    }
    return "";
}

std::string PropertyConfig::remove_from_array(std::string key, int index) {
    auto arr = get_array(key);
    if (arr.size() > index)
        return remove_from_array(key, arr.at(index));
    else
        LOG_ERROR(
            "Attempted to remove non-existent property " + key
            + " from PropertyConfig array " + key + "!");
}

StringArray PropertyConfig::remove_array(std::string key) {
    if (has_array(key)) {
        StringArray array = get_array(key);
        string_arrays.erase(key);
        string_array_keys.erase(
            std::find(string_array_keys.begin(), string_array_keys.end(), key));
        return array;
    }
    LOG_ERROR(
        "Attempted to remove non-existent property " + key
        + " from PropertyConfig!");
}

const ConfigArrayPairList PropertyConfig::get_child_arrays() const {
    ConfigArrayPairList pairs;
    for (auto key : child_array_keys)
        pairs.push_back(ConfigArrayPair(key, child_arrays.at(key)));
    return pairs;
}

const ConfigArrayPairList PropertyConfig::get_child_arrays_alphabetical() const {
    ConfigArrayPairList pairs;
    for (auto pair : child_arrays)
        pairs.push_back(ConfigArrayPair(pair.first, pair.second));
    return pairs;
}


bool PropertyConfig::has_child_array(std::string key) const
    { return child_arrays.count(key) > 0; }

const ConfigArray PropertyConfig::get_child_array(std::string key) const {
    if (has_child_array(key)) return child_arrays.at(key);
    else                return ConfigArray();
}

PropertyConfig *PropertyConfig::set_child_array(std::string key, ConfigArray array) {
    child_arrays[key] = array;
    if (std::find(child_array_keys.begin(), child_array_keys.end(), key)
            == child_array_keys.end())
        child_array_keys.push_back(key);
    return this;
}

PropertyConfig *PropertyConfig::add_to_child_array(std::string key, const PropertyConfig* config) {
    if (not has_child_array(key)) set_child_array(key, ConfigArray());
    child_arrays.at(key).push_back(new PropertyConfig(config));
    return this;
}

PropertyConfig *PropertyConfig::remove_from_child_array(
        std::string key, const PropertyConfig* config) {
    if (has_child_array(key)) {
        auto arr = child_arrays.at(key);
        auto it = std::find(arr.begin(), arr.end(), config);
        if (it != arr.end()) {
            auto to_return = *it;
            arr.erase(it);
            return to_return;
        } else
            LOG_ERROR(
                "Attempted to remove non-existent property " + key
                + " from PropertyConfig array " + key + "!");
    } else {
        LOG_ERROR(
            "Attempted to remove non-existent property " + key
            + " from PropertyConfig array " + key + "!");
    }
    return nullptr;
}

PropertyConfig *PropertyConfig::remove_from_child_array(std::string key, int index) {
    auto arr = get_child_array(key);
    if (arr.size() > index)
        return remove_from_child_array(key, arr.at(index));
    else
        LOG_ERROR(
            "Attempted to remove non-existent property " + key
            + " from PropertyConfig array " + key + "!");
}

ConfigArray PropertyConfig::remove_child_array(std::string key) {
    if (has_child_array(key)) {
        ConfigArray array = get_child_array(key);
        child_arrays.erase(key);
        child_array_keys.erase(
            std::find(child_array_keys.begin(), child_array_keys.end(), key));
        return array;
    }
    LOG_ERROR(
        "Attempted to remove non-existent property " + key
        + " from PropertyConfig!");
}


std::string PropertyConfig::str() const {
    std::string str = "{";
    for (auto pair : get_alphabetical())
        str += "(" + pair.first + "," + pair.second + ")";
    for (auto pair : get_children_alphabetical())
        str += "(" + pair.first + "," + pair.second->str() + ")";
    for (auto pair : get_arrays_alphabetical()) {
        str += "(" + pair.first + "," + "[";
        for (auto s : pair.second)
            str += s + ", ";
        str += "])";
    }
    for (auto pair : get_child_arrays_alphabetical()) {
        str += "(" + pair.first + "," + "[";
        for (auto config : pair.second)
            str += config->str() + ", ";
        str += "])";
    }
    return str + "}";
}
