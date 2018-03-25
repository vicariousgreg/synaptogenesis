#include "fixation.h"

void Fixation::update(Output* motor, OutputType output_type,
        int rows, int columns, float scale) {
    // Get current fixation cell
    int fix_row = get_y(rows);
    int fix_col = get_x(columns);

    // Compute delta based on activity
    float delta_x = 0.0;
    float delta_y = 0.0;
    float sum_activity = 0.0;

    for (int i = 0 ; i < rows ; ++i) {
        for (int j = 0 ; j < columns ; ++j) {
            float activity = (output_type == BIT)
                ? motor[i*columns + j].i & (1 << 31)
                : motor[i*columns + j].f;
            if (activity > 0.0) {
                sum_activity += activity;
                delta_y += (i - fix_row) * activity;
                delta_x += (j - fix_col) * activity;
            }
        }
    }

    if (sum_activity > 0.0) {
        delta_x = delta_x * scale / sum_activity;
        delta_y = delta_y * scale / sum_activity;

        // Update fixation
        this->x = (fix_col + delta_x) / columns;
        this->y = (fix_row + delta_y) / rows;
    }
}
