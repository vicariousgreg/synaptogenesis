#ifndef fixation_h
#define fixation_h

#include "util/constants.h"

class Fixation {
    public:
        Fixation() : x(0.5), y(0.5) { }

        float get_x() { return x; }
        float get_y() { return x; }
        int get_x(int columns) { return int(x * columns); }
        int get_y(int rows) { return int(y * rows); }

        void update(Output* motor, OutputType output_type,
            int rows, int columns, float scale);

    private:
        float x, y;
};

#endif
