#include "network/network_config.h"
#include "network/structure_config.h"
#include "network/connection_config.h"
#include "util/error_manager.h"

NetworkConfig::NetworkConfig(const PropertyConfig *config)
        : PropertyConfig(config) {
    for (auto structure : config->get_array("structures"))
        this->add_structure_internal(new StructureConfig(structure));
    for (auto connection : config->get_array("connections"))
        this->add_connection_internal(new ConnectionConfig(connection));
}


void NetworkConfig::add_structure_internal(StructureConfig* config) {
    this->structures.push_back(config);
}

NetworkConfig* NetworkConfig::add_structure(StructureConfig* config) {
    this->add_structure_internal(config);
    this->add_to_array("structures", config);
    return this;
}

NetworkConfig* NetworkConfig::add_structure(const PropertyConfig* config) {
    return this->add_structure(new StructureConfig(config));
}

const std::vector<StructureConfig*> NetworkConfig::get_structures() const
    { return structures; }


void NetworkConfig::add_connection_internal(ConnectionConfig* config) {
    this->connections.push_back(config);
}

NetworkConfig* NetworkConfig::add_connection(ConnectionConfig* config) {
    this->add_connection_internal(config);
    this->add_to_array("connections", config);
    return this;
}

NetworkConfig* NetworkConfig::add_connection(const PropertyConfig* config) {
    return this->add_connection(new ConnectionConfig(config));
}

const std::vector<ConnectionConfig*> NetworkConfig::get_connections() const
    { return connections; }
