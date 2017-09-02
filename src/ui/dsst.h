#ifndef dsst_h
#define dsst_h

#include <map>
#include <string>

#include "frontend.h"
#include "util/pointer.h"

class DSSTWindow;
class Layer;
class Environment;

class DSST : public Frontend {
    public:
        static DSST *get_instance(bool init);

        virtual ~DSST();

        void init();
        bool add_input_layer(Layer *layer, std::string params);
        bool add_output_layer(Layer *layer, std::string params);
        void update(Environment *environment);
        virtual std::string get_name() { return DSST::name; }

        Pointer<float> get_input(std::string params);
        bool is_dirty(std::string params);

        int get_input_rows();
        int get_input_columns();

        void input_symbol(int index);

    private:
        friend class DSSTWindow;

        static std::string name;
        DSST();

        Pointer<float> input_data;

        bool ui_dirty;
        bool input_dirty;
        DSSTWindow *dsst_window;
};

#endif
