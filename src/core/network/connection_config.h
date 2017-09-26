#ifndef connection_config_h
#define connection_config_h

#include "util/property_config.h"
#include "network/weight_config.h"
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

        std::string str() const;

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

        std::string str() const;

        const int row_field_size, column_field_size;
        const int row_stride, column_stride;
        const int row_offset, column_offset;
        const bool wrap;
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
            WeightConfig* weight_config);

        virtual ~ConnectionConfig();

        bool validate();

        /* Specialized config setters */
        ConnectionConfig *set_arborized_config(ArborizedConfig *config)
            { arborized_config = config; }
        ConnectionConfig *set_subset_config(SubsetConfig *config)
            { subset_config = config; }
        ConnectionConfig *set_weight_config(WeightConfig *config) {
            delete weight_config;
            weight_config = config;
        }

        /* Specialized config getters */
        ArborizedConfig *get_arborized_config() const
            { return arborized_config; }
        SubsetConfig *get_subset_config() const
            { return subset_config; }
        WeightConfig *get_weight_config() const
            { return weight_config; }

        ArborizedConfig copy_arborized_config() const {
            return (arborized_config == nullptr)
                ? ArborizedConfig() : *arborized_config;
        }
        SubsetConfig copy_subset_config() const {
            return (subset_config == nullptr)
                ? SubsetConfig() : *subset_config;
        }

        /* Setter that returns self pointer */
        ConnectionConfig *set(std::string key, std::string value) {
            set_internal(key, value);
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

    protected:
        ArborizedConfig* arborized_config;
        SubsetConfig* subset_config;
        WeightConfig* weight_config;
};

#endif
