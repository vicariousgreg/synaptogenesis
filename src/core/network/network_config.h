#ifndef network_config_h
#define network_config_h

#include "util/property_config.h"

class StructureConfig;
class ConnectionConfig;

class NetworkConfig : public PropertyConfig {
    public:
        NetworkConfig() { }
        NetworkConfig(PropertyConfig *config);

        NetworkConfig* add_structure(StructureConfig* config);
        NetworkConfig* add_structure(PropertyConfig* config);
        const std::vector<StructureConfig*> get_structures() const;

        NetworkConfig* add_connection(ConnectionConfig* config);
        NetworkConfig* add_connection(PropertyConfig* config);
        const std::vector<ConnectionConfig*> get_connections() const;

        /* Setter that returns self pointer */
        NetworkConfig *set(std::string key, std::string value) {
            PropertyConfig::set(key, value);
            return this;
        }

    protected:
        void add_structure_internal(StructureConfig* config);
        void add_connection_internal(ConnectionConfig* config);

        std::vector<StructureConfig*> structures;
        std::vector<ConnectionConfig*> connections;
};

#endif
