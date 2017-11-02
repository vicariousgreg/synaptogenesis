#include "saccade_window_impl.h"
#include "io/impl/saccade_module.h"
#include "util/tools.h"

static int peripheral_cols = 50;
static int center_cols = 100;
static int rows = 100;
static int cols = peripheral_cols * 2 + center_cols;

SaccadeWindowImpl::SaccadeWindowImpl(SaccadeModule *module)
            : module(module),
              input_dirty(true) {
    table = new Gtk::Table(1, 3, false);
    table->set_row_spacings(0);
    table->set_col_spacings(0);
    this->override_background_color(Gdk::RGBA("Black"));
    this->add(*table);

    /*
    center_cross = Gdk::Pixbuf::create_from_file(
        "./resources/saccade/center_cross.png");
    center_circle = Gdk::Pixbuf::create_from_file(
        "./resources/saccade/center_circle.png");
    peripheral_square = Gdk::Pixbuf::create_from_file(
        "./resources/saccade/peripheral_square.png");
    peripheral_circle = Gdk::Pixbuf::create_from_file(
        "./resources/saccade/peripheral_circle.png");

    // Load face images
    for (int i = 1 ; i < 10 ; ++i)
        digit_pixbufs.push_back(
            Gdk::Pixbuf::create_from_file(
                "./resources/saccade/" + std::to_string(i)
                + "_" + std::to_string(cell_cols) + ".png"));
    */

    // Create peripheral panes
    for (int i = 0; i < 2; ++i) {
        auto pix = Gdk::Pixbuf::create(
                Gdk::Colorspace::COLORSPACE_RGB,
                true, 8, peripheral_cols, rows);
        guint8* data = pix->get_pixels();

        // Clear out pane
        for (int i = 0; i < peripheral_cols*rows; ++i) {
            data[i*4 + 0] = 0;
            data[i*4 + 1] = 0;
            data[i*4 + 2] = 0;
            data[i*4 + 3] = 255;
        }

        if (i == 0) {
            this->left_pane_pixbuf = pix;
            this->left_pane_image = new Gtk::Image(pix);
        } else {
            this->right_pane_pixbuf = pix;
            this->right_pane_image = new Gtk::Image(pix);
        }
    }

    // Create center pane
    this->center_pane_pixbuf = Gdk::Pixbuf::create(
            Gdk::Colorspace::COLORSPACE_RGB,
            true, 8, center_cols, rows);
    guint8* data = this->center_pane_pixbuf->get_pixels();

    // Clear out box
    for (int i = 0; i < rows*center_cols; ++i) {
        data[i*4 + 0] = 0;
        data[i*4 + 1] = 0;
        data[i*4 + 2] = 0;
        data[i*4 + 3] = 255;
    }

    this->center_pane_image = new Gtk::Image(this->center_pane_pixbuf);

    this->table->attach(*left_pane_image, 0, 1, 0, 1);
    this->table->attach(*center_pane_image, 1, 2, 0, 1);
    this->table->attach(*right_pane_image, 2, 3, 0, 1);
    this->table->show_all();
}

SaccadeWindowImpl::~SaccadeWindowImpl() {
    delete this->table;
}

void SaccadeWindowImpl::add_layer(Layer* layer, IOTypeMask io_type) {
}

void SaccadeWindowImpl::update() {
}

void SaccadeWindowImpl::feed_input(Layer *layer, float *input) {
    if (input_dirty) {
        input_dirty = false;

        // Copy over panes
        auto pix = this->left_pane_pixbuf->get_pixels();
        for (int pix_row = 0 ; pix_row < rows ; ++pix_row)
            for (int pix_col = 0 ; pix_col < peripheral_cols ; ++pix_col)
                input[pix_row*cols + pix_col] =
                    float(pix[4*(pix_row*peripheral_cols + pix_col)]) / 255.0;

        pix = this->center_pane_pixbuf->get_pixels();
        int offset = peripheral_cols;
        for (int pix_row = 0 ; pix_row < rows ; ++pix_row)
            for (int pix_col = 0 ; pix_col < center_cols ; ++pix_col)
                input[pix_row*cols + offset + pix_col] =
                    float(pix[4*(pix_row*center_cols + pix_col)]) / 255.0;

        pix = this->right_pane_pixbuf->get_pixels();
        offset = peripheral_cols + center_cols;
        for (int pix_row = 0 ; pix_row < rows ; ++pix_row)
            for (int pix_col = 0 ; pix_col < peripheral_cols ; ++pix_col)
                input[pix_row*cols + offset + pix_col] =
                    float(pix[4*(pix_row*peripheral_cols + pix_col)]) / 255.0;
    }
}

bool SaccadeWindowImpl::on_button_press_event(GdkEventButton* button_event) {
    if (button_event->type == GDK_BUTTON_PRESS) {
        int row = int(button_event->y);
        int col = int(button_event->x);
        printf("Clicked: row(%d) column(%d)\n", row, col);
        return true;
    }
    return false;
}
