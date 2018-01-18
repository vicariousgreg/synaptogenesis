#ifndef saccade_window_impl_h
#define saccade_window_impl_h

#include "saccade_window.h"
#include "gui_window.h"
#include "util/pointer.h"

class SaccadeWindowImpl : public SaccadeWindow, public GuiWindow {
    public:
        SaccadeWindowImpl(SaccadeModule* module);
        virtual ~SaccadeWindowImpl();

        void set_face(bool fear, bool direction);
        void set_cross();

        void update();
        void add_layer(Layer *layer, IOTypeMask io_type);
        void feed_input(Layer *layer, float *input);

    protected:
        Glib::RefPtr<Gdk::Pixbuf> center_cross;
        Glib::RefPtr<Gdk::Pixbuf> center_circle;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> neutral_faces_left;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> fear_faces_left;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> neutral_faces_right;
        std::vector<Glib::RefPtr<Gdk::Pixbuf>> fear_faces_right;

        Glib::RefPtr<Gdk::Pixbuf> peripheral_square;
        Glib::RefPtr<Gdk::Pixbuf> peripheral_circle;

        Gtk::Image* left_pane_image;
        Gtk::Image* right_pane_image;
        Gtk::Image* center_pane_image;
        Glib::RefPtr<Gdk::Pixbuf> left_pane_pixbuf;
        Glib::RefPtr<Gdk::Pixbuf> right_pane_pixbuf;
        Glib::RefPtr<Gdk::Pixbuf> center_pane_pixbuf;

        bool on_button_press_event(GdkEventButton* button_event);

        bool input_dirty;
        bool window_dirty;
        bool waiting;

        Gtk::Table *table;

        SaccadeModule *module;
};

#endif
