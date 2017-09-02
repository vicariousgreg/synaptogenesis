#include "dsst_window.h"
#include "dsst.h"
#include "util/tools.h"

static const int num_rows = 8;
static const int num_cols = 18;
static const int cell_cols = 32;
static const int cell_rows = 1+2*cell_cols;
static const int spacing = 1;
static const int spacer_rows = 1;
static const int spacer_cols = 1;

DSSTWindow::DSSTWindow(DSST* dsst) : dsst(dsst), curr_prompt(0) {
    grid = new Gtk::Grid();
    grid->set_row_homogeneous(false);
    grid->set_row_spacing(spacing);
    grid->set_column_homogeneous(false);
    grid->set_column_spacing(spacing);
    grid->override_background_color(Gdk::RGBA("DarkSlateGray"));
    this->add(*grid);

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
    std::vector<Gtk::Image*> blank_cells;

    // Create blank cells for key row
    for (int col = 0; col < (num_cols/2)-1; ++col) {
        auto pix = Gdk::Pixbuf::create(
                Gdk::Colorspace::COLORSPACE_RGB,
                true, 8, cell_cols, cell_rows);
        guint8* data = pix->get_pixels();
        for (int i = 0; i < cell_cols*cell_rows; ++i) {
            data[i*4 + 0] = 0;
            data[i*4 + 1] = 0;
            data[i*4 + 2] = 0;
            data[i*4 + 3] = 0;
        }
        blank_cells.push_back(new Gtk::Image(pix));
    }

    // Create key
    for (int col = 0; col < num_cols/2; ++col) {
        auto image = key_images[col];

        if (col == 0) {
            this->grid->attach_next_to(
                *image,
                Gtk::PositionType::POS_RIGHT,
                cell_cols, cell_rows);
        } else {
            this->grid->attach_next_to(
                *image, *blank_cells[col - 1],
                Gtk::PositionType::POS_RIGHT,
                cell_cols, cell_rows);
        }

        if (col != (num_cols/2)-1)
            this->grid->attach_next_to(
                *blank_cells[col], *key_images[col],
                Gtk::PositionType::POS_RIGHT,
                cell_cols, cell_rows);
    }

    // Create spacing bar between key and prompts
    auto pix = Gdk::Pixbuf::create(
            Gdk::Colorspace::COLORSPACE_RGB,
            true, 8, spacer_cols, spacer_rows);
    auto data = pix->get_pixels();
    for (int i = 0; i < spacer_rows*spacer_cols; ++i) {
        data[i*4 + 0] = 0;
        data[i*4 + 1] = 0;
        data[i*4 + 2] = 0;
        data[i*4 + 3] = 0;
    }
    auto spacer_image = new Gtk::Image(pix);
    this->grid->attach_next_to(
        *spacer_image, *key_images[0],
        Gtk::PositionType::POS_BOTTOM,
        spacer_cols, spacer_rows);

    // Create prompt grid
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            auto image = prompt_images[row*num_cols + col];

            if (row == 0 and col == 0) {
                this->grid->attach_next_to(
                    *image, *spacer_image,
                    Gtk::PositionType::POS_BOTTOM,
                    cell_cols, cell_rows);
            } else if (col == 0) {
                this->grid->attach_next_to(
                    *image, *prompt_images[(row-1)*num_cols],
                    Gtk::PositionType::POS_BOTTOM,
                    cell_cols, cell_rows);
            } else {
                this->grid->attach_next_to(
                    *image, *prompt_images[(row)*num_cols + col - 1],
                    Gtk::PositionType::POS_RIGHT,
                    cell_cols, cell_rows);
            }
        }
    }
    this->grid->show_all();
}

DSSTWindow::~DSSTWindow() {
    delete this->grid;
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
    return cell_rows + spacing + spacer_rows + (num_rows * (cell_rows+spacing));
}

int DSSTWindow::get_input_columns() {
    return num_cols * (spacing + cell_cols) - spacing;
}

int DSSTWindow::get_input_size() {
    return get_input_rows() * get_input_columns();
}

void DSSTWindow::update_input(Pointer<float> input_data) {
    int input_cols = get_input_columns();

    // Copy over keys
    for (int key_col = 0; key_col < num_cols/2; ++key_col) {
        auto pix = this->key_pixbufs[key_col]->get_pixels();
        int col_offset = key_col * (2 * (cell_cols + spacing));

        for (int pix_row = 0 ; pix_row < cell_rows ; ++pix_row)
            for (int pix_col = 0 ; pix_col < cell_cols ; ++pix_col)
                input_data[pix_row*input_cols + col_offset + pix_col] =
                    float(pix[4*(pix_row*cell_cols + pix_col)]) / 255.0;
    }

    int prompt_offset = cell_rows + spacing + spacer_rows + spacing;

    // Copy over prompts
    for (int prompt_row = 0; prompt_row < num_rows; ++prompt_row) {
        for (int prompt_col = 0; prompt_col < num_cols; ++prompt_col) {
            auto pix = this->prompt_pixbufs[prompt_row * num_cols + prompt_col]->get_pixels();
            int row_offset = prompt_offset + prompt_row * (cell_rows + spacing);
            int col_offset = prompt_col * (cell_cols + spacing);

            for (int pix_row = 0 ; pix_row < cell_rows ; ++pix_row)
                for (int pix_col = 0 ; pix_col < cell_cols ; ++pix_col)
                    input_data[(row_offset+pix_row)*input_cols + col_offset + pix_col] =
                        float(pix[4*(pix_row*cell_cols + pix_col)]) / 255.0;
        }
    }
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
    } else if (curr_prompt == prompt_pixbufs.size()) {
        ++curr_prompt;
        int correct = 0;
        for (int i = 0 ; i < prompt_responses.size() ; ++i)
            correct += prompt_responses[i] == prompt_answers[i];
        printf("Correct: %d / %d\n", correct, prompt_responses.size());
    }
}
