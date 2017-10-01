#ifndef structure_config_h
#define structure_config_h

#include "util/property_config.h"
#include "util/constants.h"

class LayerConfig;

class StructureConfig : public PropertyConfig {
    public:
        StructureConfig(const PropertyConfig *config);
        StructureConfig(std::string name, ClusterType cluster_type);

        StructureConfig* add_layer(LayerConfig* config);
        StructureConfig* add_layer(const PropertyConfig* config);
        const std::vector<LayerConfig*> get_layers() const;

        const std::string name;
        const ClusterType cluster_type;

    protected:
        std::vector<LayerConfig*> layers;

        void add_layer_internal(LayerConfig *config);
};

#endif
