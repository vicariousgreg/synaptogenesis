#ifndef connection_config_h
#define connection_config_h

#include <string>

#include "model/weight_config.h"
#include "util/constants.h"

class ArborizedConfig {
    public:
        ArborizedConfig() : ArborizedConfig(0,0,0,0,0,0) { }
        ArborizedConfig(
            int row_field_size, int column_field_size,
            int row_stride, int column_stride,
            int row_offset=0, int column_offset=0);

        ArborizedConfig(int field_size, int stride, int offset=0);

        const int row_field_size, column_field_size;
        const int row_stride, column_stride;
        const int row_offset, column_offset;
};

class ConnectionConfig {
    public:
        ConnectionConfig(
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode,
            WeightConfig* weight_config);

        ConnectionConfig(
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode,
            WeightConfig* weight_config,
            ArborizedConfig arborized_config);

        bool plastic;
        int delay;
        float max_weight;
        ConnectionType type;
        std::string connection_params;
        ArborizedConfig arborized_config;
        WeightConfig* weight_config;
        Opcode opcode;
};

#endif
