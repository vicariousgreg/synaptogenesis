#include "dsst_window.h"
#include "dsst.h"
#include "util/tools.h"

static const int num_rows = 8;
static const int num_cols = 18;
static const int cell_cols = 32;
static const int cell_rows = 1+2*cell_cols;
static const int spacing = cell_cols/4;

DSSTWindow::DSSTWindow(DSST* dsst) : dsst(dsst), curr_prompt(0) {
    table = new Gtk::Table(num_rows + 2, num_cols, false);
    table->set_row_spacings(spacing);
    table->set_col_spacings(spacing);
    this->override_background_color(Gdk::RGBA("Black"));
    this->add(*table);

    // Load images
    for (int i = 1 ; i < 10 ; ++i)
        digit_pixbufs.push_back(
            Gdk::Pixbuf::create_from_file(
                "./resources/dsst/" + std::to_string(i)
                + "_" + std::to_string(cell_cols) + ".png"));
    for (int i = 1 ; i < 10 ; ++i)
        symbol_pixbufs.push_back(
            Gdk::Pixbuf::create_from_file(
                "./resources/dsst/sym" + std::to_string(i)
                + "_" + std::to_string(cell_cols) + ".png"));

    // Create key boxes
    for (int col = 0; col < num_cols/2; ++col) {
        auto pix = Gdk::Pixbuf::create(
                Gdk::Colorspace::COLORSPACE_RGB,
                true, 8, cell_cols, cell_rows);
        guint8* data = pix->get_pixels();

        // Clear out box
        for (int i = 0; i < cell_cols*cell_rows; ++i) {
            data[i*4 + 0] = 0;
            data[i*4 + 1] = 0;
            data[i*4 + 2] = 0;
            data[i*4 + 3] = 255;
        }

        // Add center divider
        for (int i = 2; i < cell_cols-2; ++i) {
            data[(cell_cols*cell_cols+i)*4 + 0] = 255;
            data[(cell_cols*cell_cols+i)*4 + 1] = 255;
            data[(cell_cols*cell_cols+i)*4 + 2] = 255;
        }

        this->key_pixbufs.push_back(pix);
        this->key_images.push_back(new Gtk::Image(pix));
        add_digit(col+1, pix);
        add_symbol(col+1, pix);
    }

    // Create prompt boxes
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            auto pix = Gdk::Pixbuf::create(
                    Gdk::Colorspace::COLORSPACE_RGB,
                    true, 8, cell_cols, cell_rows);
            guint8* data = pix->get_pixels();

            // Clear out box
            for (int i = 0; i < cell_cols*cell_rows; ++i) {
                data[i*4 + 0] = 0;
                data[i*4 + 1] = 0;
                data[i*4 + 2] = 0;
                data[i*4 + 3] = 255;
            }

            // Add center divider
            for (int i = 2; i < cell_cols-2; ++i) {
                data[(cell_cols*cell_cols+i)*4 + 0] = 255;
                data[(cell_cols*cell_cols+i)*4 + 1] = 255;
                data[(cell_cols*cell_cols+i)*4 + 2] = 255;
            }
            this->prompt_pixbufs.push_back(pix);
            this->prompt_images.push_back(new Gtk::Image(pix));

            int index = iRand(1,9);
            add_digit(index, pix);
            prompt_answers.push_back(index);
            dirty.push_back(false);
        }
    }
}

void DSSTWindow::init() {
    // Create key
    for (int col = 0; col < num_cols/2; ++col)
        this->table->attach(
            *key_images[col], col*2, col*2+1, 0, 1);

    // Spacer
    auto spacer_pix = Gdk::Pixbuf::create(
            Gdk::Colorspace::COLORSPACE_RGB,
            true, 8, cell_cols, cell_rows);
    auto spacer = new Gtk::Image(spacer_pix);
    this->table->attach(*spacer, 0, num_cols, 1, 2);

    // Create prompt grid
    for (int row = 0; row < num_rows; ++row)
        for (int col = 0; col < num_cols; ++col)
            this->table->attach(
                *prompt_images[row*num_cols + col],
                col, col+1, row+2, row+3);
    this->table->show_all();
}

DSSTWindow::~DSSTWindow() {
    delete this->table;
}

void DSSTWindow::add_layer(LayerInfo* layer_info) {
}

void DSSTWindow::update() {
    for (int i = 0; i < dirty.size(); ++i)
        if (dirty[i]) {
            prompt_images[i]->set(prompt_pixbufs[i]);
            dirty[i] = false;
        }
}

int DSSTWindow::get_input_rows() {
    return (num_rows + 2) * (cell_rows + spacing) - spacing;
}

int DSSTWindow::get_input_columns() {
    return num_cols * (spacing + cell_cols) - spacing;
}

int DSSTWindow::get_input_size() {
    return get_input_rows() * get_input_columns();
}

void DSSTWindow::update_input() {
    float* input = dsst->input_data;
    int input_cols = get_input_columns();

    // Copy over keys
    for (int key_col = 0; key_col < num_cols/2; ++key_col) {
        auto pix = this->key_pixbufs[key_col]->get_pixels();
        int col_offset = key_col * (2 * (cell_cols + spacing));

        for (int pix_row = 0 ; pix_row < cell_rows ; ++pix_row)
            for (int pix_col = 0 ; pix_col < cell_cols ; ++pix_col)
                input[pix_row*input_cols + col_offset + pix_col] =
                    float(pix[4*(pix_row*cell_cols + pix_col)]) / 255.0;
    }

    int prompt_offset = 2 * (cell_rows + spacing);

    // Copy over prompts
    for (int prompt_row = 0; prompt_row < num_rows; ++prompt_row) {
        for (int prompt_col = 0; prompt_col < num_cols; ++prompt_col) {
            auto pix = this->prompt_pixbufs[prompt_row * num_cols + prompt_col]->get_pixels();
            int row_offset = prompt_offset + prompt_row * (cell_rows + spacing);
            int col_offset = prompt_col * (cell_cols + spacing);

            for (int pix_row = 0 ; pix_row < cell_rows ; ++pix_row)
                for (int pix_col = 0 ; pix_col < cell_cols ; ++pix_col)
                    input[(row_offset+pix_row)*input_cols + col_offset + pix_col] =
                        float(pix[4*(pix_row*cell_cols + pix_col)]) / 255.0;
        }
    }

    dsst->input_dirty = true;
}

void DSSTWindow::add_digit(int index, Glib::RefPtr<Gdk::Pixbuf> pix) {
    if (index < 1 or index > 9)
        ErrorManager::get_instance()->log_error(
            "Attempted to add out of bounds digit to DSST!");

    auto digit_data = digit_pixbufs[index-1]->get_pixels();
    auto box_data = pix->get_pixels();
    int pix_size = digit_pixbufs[index-1]->get_has_alpha() ? 4 : 3;

    for (int index = 0; index < cell_cols*cell_cols; ++index) {
        box_data[4*index + 0] = digit_data[pix_size*index + 0];
        box_data[4*index + 1] = digit_data[pix_size*index + 1];
        box_data[4*index + 2] = digit_data[pix_size*index + 2];
    }
}

void DSSTWindow::add_symbol(int index, Glib::RefPtr<Gdk::Pixbuf> pix) {
    if (index < 1 or index > 9)
        ErrorManager::get_instance()->log_error(
            "Attempted to add out of bounds symbol to DSST!");

    auto symbol_data = symbol_pixbufs[index-1]->get_pixels();
    auto box_data = pix->get_pixels();
    int offset = cell_cols * (1+cell_cols);
    int pix_size = symbol_pixbufs[index-1]->get_has_alpha() ? 4 : 3;

    for (int index = 0; index < cell_cols*cell_cols; ++index) {
        box_data[4*(offset+index) + 0] = symbol_data[pix_size*index + 0];
        box_data[4*(offset+index) + 1] = symbol_data[pix_size*index + 1];
        box_data[4*(offset+index) + 2] = symbol_data[pix_size*index + 2];
    }
}

void DSSTWindow::input_symbol(int index) {
    if (curr_prompt < prompt_pixbufs.size()) {
        add_symbol(index, prompt_pixbufs[curr_prompt]);
        prompt_responses.push_back(index);
        dirty[curr_prompt++] = true;
        update_input();
    } else if (curr_prompt == prompt_pixbufs.size()) {
        ++curr_prompt;
        int correct = 0;
        for (int i = 0 ; i < prompt_responses.size() ; ++i)
            correct += prompt_responses[i] == prompt_answers[i];
        printf("Correct: %d / %d\n", correct, prompt_responses.size());
    }
}

bool DSSTWindow::on_button_press_event(GdkEventButton* button_event) {
    if (button_event->type == GDK_BUTTON_PRESS) {
        int row = int(button_event->y) / cell_rows;
        int col = int(button_event->x) / (cell_cols + spacing);

        if (row == 0 and not (col % 2))
            input_symbol(1 + (col / 2));

        return true;
    }
    return false;
}
