#ifndef connection_config_h
#define connection_config_h

#include "util/property_config.h"
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
        ArborizedConfig();

        int get_total_field_size() const
            { return row_field_size * column_field_size; }

        int is_regular() const {
            return (row_stride == column_stride == 1)
                and (row_field_size == column_field_size)
                and (row_offset == column_offset == -row_field_size/2)
                and (row_spacing == column_spacing == 1);
        }

        PropertyConfig to_property_config() const;

        std::string str() const;

        bool validate(Connection *conn) const;

        int row_field_size, column_field_size;
        int row_stride, column_stride;
        int row_offset, column_offset;
        int row_spacing, column_spacing;
        bool wrap;
};

class ConnectionConfig : public PropertyConfig {
    public:
        ConnectionConfig(const PropertyConfig *config);

        ConnectionConfig(
            std::string from_layer,
            std::string to_layer,
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode,
            bool sparse=false,
            bool randomized_projection=false,
            bool convolutional=false,
            bool recurrent=false,
            PropertyConfig *specialized_config=nullptr,
            PropertyConfig *weight_config=nullptr,
            std::string dendrite="root",
            std::string name="");

        bool validate(Connection *conn) const;

        /* Specialized config getters */
        const ArborizedConfig get_arborized_config() const;
        const SubsetConfig get_subset_config() const;
        const PropertyConfig get_weight_config() const;

        std::string str() const;

        const std::string name;
        const std::string from_layer;
        const std::string to_layer;
        const std::string dendrite;
        const bool plastic;
        const int delay;
        const float max_weight;
        const ConnectionType type;
        const Opcode opcode;
        const bool sparse;
        const bool randomized_projection;
        const bool convolutional;
        const bool recurrent;
};

#endif
