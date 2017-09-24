#ifndef dsst_window_impl_h
#define dsst_window_impl_h

#include "dsst_window.h"
#include "gui_window.h"
#include "util/pointer.h"

class DSSTWindowImpl : public DSSTWindow, public GuiWindow {
    public:
        DSSTWindowImpl(DSSTModule* module);
        virtual ~DSSTWindowImpl();

        void update();
        void add_layer(Layer *layer, IOTypeMask io_type);
        void feed_input(Layer *layer, float *input);
        void input_symbol(int index);

    protected:
        std::vector<Gtk::Image*> key_images;
        std::vector<Gtk::Image*> prompt_images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> key_pixbufs;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> prompt_pixbufs;

        std::vector<Glib::RefPtr<Gdk::Pixbuf>> digit_pixbufs;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> symbol_pixbufs;
        std::vector<int> prompt_answers;
        std::vector<int> prompt_responses;
        std::vector<bool> dirty;

        void add_digit(int index, Glib::RefPtr<Gdk::Pixbuf> pix);
        void add_symbol(int index, Glib::RefPtr<Gdk::Pixbuf> pix);
        bool on_button_press_event(GdkEventButton* button_event);

        int curr_prompt;
        bool input_dirty;

        Gtk::Table *table;

        DSSTModule *module;

        const int num_rows, num_cols,
            cell_cols, cell_rows,
            spacing, input_columns;
};

#endif
