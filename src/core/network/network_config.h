#ifndef network_config_h
#define network_config_h

#include "util/property_config.h"

class StructureConfig;
class ConnectionConfig;

class NetworkConfig : public PropertyConfig {
    public:
        NetworkConfig() { }
        NetworkConfig(const PropertyConfig *config);

        NetworkConfig* add_structure(StructureConfig* config);
        NetworkConfig* add_structure(const PropertyConfig* config);
        const std::vector<StructureConfig*> get_structures() const;

        NetworkConfig* add_connection(ConnectionConfig* config);
        NetworkConfig* add_connection(const PropertyConfig* config);
        const std::vector<ConnectionConfig*> get_connections() const;

    protected:
        void add_structure_internal(StructureConfig* config);
        void add_connection_internal(ConnectionConfig* config);

        std::vector<StructureConfig*> structures;
        std::vector<ConnectionConfig*> connections;
};

#endif
