#ifndef connection_config_h
#define connection_config_h

#include <string>

#include "model/weight_config.h"
#include "util/constants.h"

class ConnectionConfig {
    public:
        ConnectionConfig(
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode,
            std::string connection_params,
            WeightConfig* weight_config);

        bool plastic;
        int delay;
        float max_weight;
        ConnectionType type;
        std::string connection_params;
        WeightConfig* weight_config;
        Opcode opcode;
};

class ArborizedConfig {
    public:
        ArborizedConfig(
            int row_field_size, int column_field_size,
            int row_stride, int column_stride,
            int row_offset=0, int column_offset=0);

        ArborizedConfig(int field_size, int stride, int offset=0);

        operator std::string() const;

        static ArborizedConfig decode(std::string params);

        const int row_field_size, column_field_size;
        const int row_stride, column_stride;
        const int row_offset, column_offset;
};

#endif
