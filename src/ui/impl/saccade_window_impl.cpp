#include "saccade_window_impl.h"
#include "io/impl/saccade_module.h"
#include "util/tools.h"

static int peripheral_cols = 120;
static int center_cols = 240;
static int rows = 340;
static int cols = peripheral_cols * 2 + center_cols;

SaccadeWindowImpl::SaccadeWindowImpl(SaccadeModule *module)
            : module(module),
              window_dirty(true),
              input_dirty(true),
              waiting(true) {
    // Add table
    table = new Gtk::Table(1, 3, false);
    table->set_row_spacings(0);
    table->set_col_spacings(0);
    this->override_background_color(Gdk::RGBA("Black"));

    // Add Overlay
    overlay = new Gtk::Overlay();
    overlay->add(*table);
    overlay->set_opacity(1.0);
    this->overlay_pix = Gdk::Pixbuf::create(
        Gdk::Colorspace::COLORSPACE_RGB, true, 8, cols, rows);

    // Transparent red image
    guint8* data = overlay_pix->get_pixels();
    for (int i = 0 ; i < rows*cols ; ++i) {
        data[4*i + 0] = 255;
        data[4*i + 1] = 0;
        data[4*i + 2] = 0;
        data[4*i + 3] = 0;
    }
    this->overlay_image = new Gtk::Image(overlay_pix);
    overlay->add_overlay(*overlay_image);
    this->add(*overlay);

    try {
        center_cross = Gdk::Pixbuf::create_from_file(
            "./resources/antisaccade/center_cross.bmp");
        center_circle = Gdk::Pixbuf::create_from_file(
            "./resources/antisaccade/center_circle.bmp");
        peripheral_square = Gdk::Pixbuf::create_from_file(
            "./resources/antisaccade/peripheral_square.bmp");
        peripheral_circle = Gdk::Pixbuf::create_from_file(
            "./resources/antisaccade/peripheral_circle.bmp");

        // Load face images
        for (int i = 0 ; i < 9 ; ++i) {
            fear_faces_left.push_back(
                Gdk::Pixbuf::create_from_file(
                    "./resources/antisaccade/fear_l_" + std::to_string(i) + ".bmp"));
            fear_faces_right.push_back(
                Gdk::Pixbuf::create_from_file(
                    "./resources/antisaccade/fear_r_" + std::to_string(i) + ".bmp"));
        }
        for (int i = 0 ; i < 8 ; ++i) {
            neutral_faces_left.push_back(
                Gdk::Pixbuf::create_from_file(
                    "./resources/antisaccade/neutral_l_" + std::to_string(i) + ".bmp"));
            neutral_faces_right.push_back(
                Gdk::Pixbuf::create_from_file(
                    "./resources/antisaccade/neutral_r_" + std::to_string(i) + ".bmp"));
        }
    } catch(Glib::FileError) {
        LOG_ERROR("Could not open resources directory!");
    }

    // Create peripheral panes
    auto pix = peripheral_square;
    data = pix->get_pixels();

    this->left_pane_pixbuf = pix;
    this->left_pane_image = new Gtk::Image(pix);
    this->right_pane_pixbuf = pix;
    this->right_pane_image = new Gtk::Image(pix);

    // Create center pane
    this->center_pane_pixbuf = Gdk::Pixbuf::create(
            Gdk::Colorspace::COLORSPACE_RGB,
            true, 8, center_cols, rows);
    data = this->center_pane_pixbuf->get_pixels();

    // Clear out box
    for (int i = 0; i < rows*center_cols; ++i) {
        data[i*4 + 0] = 255;
        data[i*4 + 1] = 255;
        data[i*4 + 2] = 255;
        data[i*4 + 3] = 255;
    }

    this->center_pane_image = new Gtk::Image(this->center_pane_pixbuf);

    this->table->attach(*left_pane_image, 0, 1, 0, 1);
    this->table->attach(*center_pane_image, 1, 2, 0, 1);
    this->table->attach(*right_pane_image, 2, 3, 0, 1);
    this->table->show_all();
    this->overlay->show_all();

    this->set_cross();
}

SaccadeWindowImpl::~SaccadeWindowImpl() {
    delete this->table;
    delete this->overlay;
}

void SaccadeWindowImpl::set_cross() {
    auto cross_data = center_cross->get_pixels();
    int pix_size = center_cross->get_has_alpha() ? 4 : 3;
    auto center_data = center_pane_pixbuf->get_pixels();

    for (int index = 0; index < rows*center_cols; ++index) {
        center_data[4*index + 0] = cross_data[pix_size*index + 0];
        center_data[4*index + 1] = cross_data[pix_size*index + 1];
        center_data[4*index + 2] = cross_data[pix_size*index + 2];
    }

    input_dirty = true;
    window_dirty = true;
    waiting = true;
}

void SaccadeWindowImpl::set_face(bool fear, bool direction) {
    auto& faces = fear_faces_left;
    if (fear and direction)
        faces = fear_faces_right;
    else if (not fear and direction)
        faces = neutral_faces_right;
    else if (not fear and not direction)
        faces = neutral_faces_left;

    int index = iRand(faces.size()-1);
    auto face_pix = faces[index];
    auto face_data = face_pix->get_pixels();
    int pix_size = face_pix->get_has_alpha() ? 4 : 3;
    auto center_data = center_pane_pixbuf->get_pixels();

    printf("Setting face to ");
    printf((fear) ? "fear " : "neutral ");
    printf((direction) ? "right " : "left ");
    printf("%d\n", index);

    for (int index = 0; index < rows*center_cols; ++index) {
        center_data[4*index + 0] = face_data[pix_size*index + 0];
        center_data[4*index + 1] = face_data[pix_size*index + 1];
        center_data[4*index + 2] = face_data[pix_size*index + 2];
    }

    input_dirty = true;
    window_dirty = true;
    waiting = false;
}

void SaccadeWindowImpl::add_layer(Layer* layer, IOTypeMask io_type) {
}

void SaccadeWindowImpl::update() {
    if (window_dirty) {
        left_pane_image->set(left_pane_pixbuf);
        right_pane_image->set(right_pane_pixbuf);
        center_pane_image->set(center_pane_pixbuf);
        overlay_image->set(overlay_pix);
        window_dirty = false;
    }
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

void SaccadeWindowImpl::report_output(Layer *layer,
        Output *output, OutputType output_type) {
    int old_col = fixation.get_x(cols);
    int old_row = fixation.get_y(rows);

    fixation.update(output, output_type, layer->rows, layer->columns, 1.0);

    int col = fixation.get_x(cols);
    int row = fixation.get_y(rows);

    if (row != old_row or col != old_col) {
        //printf("Fixation: %d %d\n", row, col);
        auto data = this->overlay_pix->get_pixels();
        for (int i = 0 ; i < rows*cols ; ++i)
            data[4*i + 3] = 0;

        int start_row = std::max(0, row-5);
        int end_row = std::min(rows, row+5);
        int start_col = std::max(0, col-5);
        int end_col = std::min(cols, col+5);

        for (int i = start_row ; i < end_row ; ++i) {
            for (int j = start_col ; j < end_col ; ++j) {
                int index = i * cols + j;
                data[4*index + 3] = 255;
            }
        }
        window_dirty = true;
    }
}

bool SaccadeWindowImpl::on_button_press_event(GdkEventButton* button_event) {
    if (button_event->type == GDK_BUTTON_PRESS) {
        int row = int(button_event->y);
        int col = int(button_event->x);
        if (col < peripheral_cols) {
            printf("Clicked left   (%d, %d)\n", row, col);
            if (not waiting) this->set_cross();
        } else if (col < peripheral_cols + center_cols) {
            printf("Clicked center (%d, %d)\n", row, col);
            if (waiting) this->set_face(iRand(1), iRand(1));
        } else {
            printf("Clicked right  (%d, %d)\n", row, col);
            if (not waiting) this->set_cross();
        }
        return true;
    }
    return false;
}
