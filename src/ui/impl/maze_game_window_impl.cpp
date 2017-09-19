#include "maze_game_window_impl.h"
#include "io/impl/maze_module.h"

#include "network/structure.h"

MazeGameWindowImpl::MazeGameWindowImpl() {
    grid = new Gtk::Grid();
    this->add(*grid);
    cell_size = 20;

    board_dim = MazeModule::board_dim;

    for (int row = 0; row < board_dim; ++row) {
        for (int col = 0; col < board_dim; ++col) {
            auto pix = Gdk::Pixbuf::create(
                    Gdk::Colorspace::COLORSPACE_RGB,
                    true, 8, cell_size, cell_size);
            guint8* data = pix->get_pixels();
            for (int i = 0; i < cell_size*cell_size; ++i) {
                data[i*4 + 0] = 0;
                data[i*4 + 1] = 0;
                data[i*4 + 2] = 0;
                data[i*4 + 3] = 255;
            }
            auto image = new Gtk::Image(pix);

            this->pixbufs.push_back(pix);
            this->images.push_back(image);

            if (row == 0 and col == 0) {
                this->grid->attach_next_to(
                    *image,
                    Gtk::PositionType::POS_RIGHT,
                    cell_size, cell_size);
            } else if (col == 0) {
                this->grid->attach_next_to(
                    *image, *images[(row-1)*board_dim],
                    Gtk::PositionType::POS_BOTTOM,
                    cell_size, cell_size);
            } else {
                this->grid->attach_next_to(
                    *image, *images[(row)*board_dim + col - 1],
                    Gtk::PositionType::POS_RIGHT,
                    cell_size, cell_size);
            }
        }
    }
}

MazeGameWindowImpl::~MazeGameWindowImpl() {
    delete this->grid;
}

void MazeGameWindowImpl::set_cell(int row, int col, int r, int g, int b) {
    guint8* data = pixbufs[row*board_dim + col]->get_pixels();
    for (int i = 0; i < cell_size*cell_size; ++i) {
        data[i*4 + 0] = r;
        data[i*4 + 1] = g;
        data[i*4 + 2] = b;
    }
}

void MazeGameWindowImpl::add_layer(Layer *layer, IOTypeMask io_type) { }

void MazeGameWindowImpl::update() {
    for (int i = 0; i < images.size(); ++i)
        images[i]->set(this->pixbufs[i]);
}

void MazeGameWindowImpl::set_cell_clear(int row, int col) {
    this->set_cell(row, col, 0, 0, 0);
}

void MazeGameWindowImpl::set_cell_player(int row, int col) {
    this->set_cell(row, col, 255, 0, 0);
}

void MazeGameWindowImpl::set_cell_goal(int row, int col) {
    this->set_cell(row, col, 0, 255, 0);
}
