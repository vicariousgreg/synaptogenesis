#ifndef connection_config_h
#define connection_config_h

#include "model/property_config.h"
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

class ConnectionConfig : public PropertyConfig {
    public:
        ConnectionConfig(
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode,
            WeightConfig* weight_config,
            ArborizedConfig* arborized_config = new ArborizedConfig());

        virtual ~ConnectionConfig();

        /* Setter that returns self pointer */
        ConnectionConfig *set_property(std::string key, std::string value) {
            set_property_internal(key, value);
            return this;
        }

        /* Gets the expected row/col size of a destination layer given.
         * This function only returns meaningful values for connection types that
         *   are not FULLY_CONNECTED, because they can link layers of any sizes */
        int get_expected_rows(int from_rows);
        int get_expected_columns(int from_columns);

        const bool plastic;
        const int delay;
        const float max_weight;
        const ConnectionType type;
        const Opcode opcode;
        WeightConfig* const weight_config;
        ArborizedConfig* const arborized_config;
};

#endif
