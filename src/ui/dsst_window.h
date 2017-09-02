#ifndef dsst_window_h
#define dsst_window_h

#include "gui_window.h"
#include "util/pointer.h"

class DSST;

class DSSTWindow : public GuiWindow {
    public:
        DSSTWindow(DSST *dsst);
        virtual ~DSSTWindow();

        void add_layer(LayerInfo *layer_info);
        void init();
        void update();

        int get_input_rows();
        int get_input_columns();
        int get_input_size();

        void update_input();
        void input_symbol(int index);

        virtual bool on_button_press_event(GdkEventButton* button_event);

    private:
        std::vector<Gtk::Image*> key_images;
        std::vector<Gtk::Image*> prompt_images;
        std::vector<Glib::RefPtr<Gdk::Pixbuf> > key_pixbufs;
        std::vector<Glib::RefPtr<Gdk::Pixbuf> > prompt_pixbufs;

        std::vector<Glib::RefPtr<Gdk::Pixbuf> > digit_pixbufs;
        std::vector<Glib::RefPtr<Gdk::Pixbuf> > symbol_pixbufs;
        std::vector<int> prompt_answers;
        std::vector<int> prompt_responses;
        std::vector<bool> dirty;

        void add_digit(int index, Glib::RefPtr<Gdk::Pixbuf> pix);
        void add_symbol(int index, Glib::RefPtr<Gdk::Pixbuf> pix);

        int curr_prompt;

        DSST* dsst;
        Gtk::Table *table;
};

#endif
