#ifndef connection_config_h
#define connection_config_h

#include "model/property_config.h"
#include "model/weight_config.h"
#include "util/constants.h"

class Connection;

class SubsetConfig {
    public:
        SubsetConfig() : SubsetConfig(0,0,0,0,0,0,0,0) { }
        SubsetConfig(
            int from_row_start, int from_row_end,
            int from_col_start, int from_col_end,
            int to_row_start, int to_row_end,
            int to_col_start, int to_col_end);

        bool validate(Connection *conn);

        const int from_row_start, from_row_end;
        const int from_col_start, from_col_end;
        const int from_row_size, from_col_size;
        const int from_size;
        const int to_row_start, to_row_end;
        const int to_col_start, to_col_end;
        const int to_row_size, to_col_size;
        const int to_size;
        const int total_size;
};

class ArborizedConfig {
    public:
        ArborizedConfig() : ArborizedConfig(0,0,0,0,0,0) { }

        ArborizedConfig(
            int row_field_size, int column_field_size,
            int row_stride, int column_stride);
        ArborizedConfig(
            int row_field_size, int column_field_size,
            int row_stride, int column_stride,
            int row_offset, int column_offset);

        ArborizedConfig(int field_size, int stride=1);
        ArborizedConfig(int field_size, int stride, int offset);

        int get_total_field_size() const
            { return row_field_size * column_field_size; }

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
            WeightConfig* weight_config);

        virtual ~ConnectionConfig();

        /* Specialized config setters */
        ConnectionConfig *set_arborized_config(ArborizedConfig *config)
            { arborized_config = config; }
        ConnectionConfig *set_subset_config(SubsetConfig *config)
            { subset_config = config; }

        /* Specialized config getters */
        ArborizedConfig *get_arborized_config() const
            { return arborized_config; }
        SubsetConfig *get_subset_config() const
            { return subset_config; }

        ArborizedConfig copy_arborized_config() const {
            return (arborized_config == nullptr)
                ? ArborizedConfig() : *arborized_config;
        }
        SubsetConfig copy_subset_config() const {
            return (subset_config == nullptr)
                ? SubsetConfig() : *subset_config;
        }

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

    protected:
        ArborizedConfig* arborized_config;
        SubsetConfig* subset_config;
};

#endif
