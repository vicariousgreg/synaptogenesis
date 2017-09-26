#ifndef connection_config_h
#define connection_config_h

#include "util/property_config.h"
#include "network/weight_config.h"
#include "util/constants.h"

class Connection;

class SubsetConfig {
    public:
        SubsetConfig(PropertyConfig *config);
        SubsetConfig() : SubsetConfig(0,0,0,0,0,0,0,0) { }
        SubsetConfig(
            int from_row_start, int from_row_end,
            int from_col_start, int from_col_end,
            int to_row_start, int to_row_end,
            int to_col_start, int to_col_end);

        bool validate(Connection *conn) const;
        PropertyConfig to_property_config() const;

        std::string str() const;

        int from_row_start, from_row_end;
        int from_col_start, from_col_end;
        int from_row_size, from_col_size;
        int from_size;
        int to_row_start, to_row_end;
        int to_col_start, to_col_end;
        int to_row_size, to_col_size;
        int to_size;
        int total_size;
};

class ArborizedConfig {
    public:
        ArborizedConfig(PropertyConfig *config);
        ArborizedConfig() : ArborizedConfig(0,0,0,0,0,0) { }

        ArborizedConfig(
            int row_field_size, int column_field_size,
            int row_stride, int column_stride,
            bool wrap=false);
        ArborizedConfig(
            int row_field_size, int column_field_size,
            int row_stride, int column_stride,
            int row_offset, int column_offset,
            bool wrap=false);

        ArborizedConfig(int field_size, int stride=1, bool wrap=false);
        ArborizedConfig(int field_size, int stride, int offset, bool wrap=false);

        int get_total_field_size() const
            { return row_field_size * column_field_size; }

        int is_regular() const {
            return (row_stride == column_stride == 1)
                and (row_field_size == column_field_size)
                and (row_offset == column_offset == -row_field_size/2);
        }

        PropertyConfig to_property_config() const;

        std::string str() const;

        int row_field_size, column_field_size;
        int row_stride, column_stride;
        int row_offset, column_offset;
        bool wrap;
};

class ConnectionConfig : public PropertyConfig {
    public:
        ConnectionConfig(PropertyConfig *config);

        ConnectionConfig(
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode);

        ConnectionConfig(
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode,
            PropertyConfig* weight_config);

        virtual ~ConnectionConfig();

        bool validate(Connection *conn) const;

        /* Specialized config setters */
        ConnectionConfig *set_arborized_config(ArborizedConfig *config);
        ConnectionConfig *set_subset_config(SubsetConfig *config);

        ConnectionConfig *set_arborized_config(PropertyConfig *config);
        ConnectionConfig *set_subset_config(PropertyConfig *config);
        ConnectionConfig *set_weight_config(PropertyConfig *config);

        /* Specialized config getters */
        const ArborizedConfig get_arborized_config() const;
        const SubsetConfig get_subset_config() const;
        const PropertyConfig get_weight_config() const;

        /* Setter that returns self pointer */
        ConnectionConfig *set(std::string key, std::string value) {
            set_value(key, value);
            return this;
        }

        std::string str() const;

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
};

#endif
