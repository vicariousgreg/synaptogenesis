#ifndef saccade_window_impl_h
#define saccade_window_impl_h

#include <mutex>

#include "saccade_window.h"
#include "gui_window.h"
#include "fixation.h"
#include "util/resources/pointer.h"

class SaccadeWindowImpl : public SaccadeWindow, public GuiWindow {
    public:
        SaccadeWindowImpl(SaccadeModule* module);
        virtual ~SaccadeWindowImpl();

        void lock() { mutex.lock(); }
        void unlock() { mutex.unlock(); }

        void set_face();
        void set_face(bool fear, bool direction, int face_index=-1);
        void set_cross();

        void update();
        void add_layer(Layer *layer, IOTypeMask io_type);
        void prepare_input_data();
        void feed_input(Layer *layer, float *input);
        void feed_central_input(Layer *layer, float *input);
        void report_output(Layer *layer,
            Output *output, OutputType output_type);
        void update_fixation_point(int row, int col);

    protected:
        Glib::RefPtr<Gdk::Pixbuf> overlay_pix;
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
        Gtk::Image* overlay_image;
        Glib::RefPtr<Gdk::Pixbuf> left_pane_pixbuf;
        Glib::RefPtr<Gdk::Pixbuf> right_pane_pixbuf;
        Glib::RefPtr<Gdk::Pixbuf> center_pane_pixbuf;

        bool on_button_press_event(GdkEventButton* button_event);
        void click_center();
        void click_peripheral();

        std::mutex mutex;
        bool input_dirty;
        bool central_input_dirty;
        bool window_dirty;

        Gtk::Table *table;
        Gtk::Overlay *overlay;

        SaccadeModule *module;

        Fixation fixation;
        float saccade_rate;

        float* input_data;

        bool shuffle;
        bool automatic;
        int num_faces;
        int cross_time;
        int face_time;
        int last_face_time;
        int iteration;
        int counter;

        bool face_on;
        bool curr_fear;
        bool curr_direction;
        bool curr_face_index;
};

#endif
