import os
import struct
import sys
import gtk

class Layer:
    def __init__(self, input_index, output_index, rows, columns, is_input, is_output, out_type):
        self.output = [0] * (rows * columns)
        if out_type == "float":
            self.print_out = self.print_float
        elif out_type == "int":
            self.print_out = self.print_int
        elif out_type == "bit":
            self.print_out = self.print_bit
        else: raise

        self.input_index = input_index
        self.output_index = output_index
        self.rows = rows
        self.columns = columns
        self.is_input = is_input
        self.is_output = is_output
        self.size = rows * columns

        print("Layer: (input_index=%d, output_index=%d, rows=%d, "
                      "cols=%d, is_input=%s, is_output=%s)" %
                (input_index, output_index, rows,
                 columns, is_input, is_output))

    def print_float(self):
        print("=" * 80)
        out = ""
        for i in xrange(self.rows):
            for j in xrange(self.columns):
                index = i * self.columns + j
                val = self.output[index]

                if (val > 0.75):    out += " X";
                elif (val > 0.65):  out += " @";
                elif (val > 0.50):  out += " +";
                elif (val > 0.25): out += " *";
                elif (val > 0.10): out += " -";
                else:               out += " '";
            out += "\n"
        print(out)

    def print_int(self):
        pass

    def print_bit(self):
        print("=" * 80)
        chars = "XXXXXX@@@@@@@@@++++++++++-----------''''............"
        out = ""
        for i in xrange(self.rows):
            for j in xrange(self.columns):
                index = i * self.columns + j
                val = self.output[index]
                if val == 0:
                    out += "  "
                else:
                    counter = 0
                    mask = 1
                    while val & mask == 0:
                        mask *= 2
                        counter += 1
                    if counter < len(chars):
                        out += chars[counter] + " "
                    else:
                        out += "  "
            out += "\n"
        print(out)

class PyApp(gtk.Window):
    def __init__(self):
        super(PyApp, self).__init__()
        self.set_title("Buttons")
        self.set_size_request(250, 200)
        self.set_position(gtk.WIN_POS_CENTER)

        btn1 = gtk.Button("Button")
        btn1.set_sensitive(False)
        btn2 = gtk.Button("Button")
        btn3 = gtk.Button(stock=gtk.STOCK_CLOSE)
        btn4 = gtk.Button("Button")
        btn4.set_size_request(80, 40)

        fixed = gtk.Fixed()

        fixed.put(btn1, 20, 30)
        fixed.put(btn2, 100, 30)
        fixed.put(btn3, 20, 80)
        fixed.put(btn4, 100, 80)

        self.connect("destroy", gtk.main_quit)

        self.add(fixed)
        self.show_all()

def read_val(f, fmt):
    line = os.read(f, 4)
    if len(line) > 0:
        return struct.unpack(fmt, line)[0]
    else: raise OSError

def read_vals(f, fmt, count, dest):
    line = os.read(f, count * 4)
    if len(line) > 0:
        for i in xrange(count):
            dest[i] = struct.unpack_from(fmt, line, i * 4)[0]
    else: raise OSError

def main(fifo_name, out_type):
    if out_type == "float":
        fmt = 'f'
    elif out_type == "int":
        fmt = 'i'
    elif out_type == "bit":
        fmt = 'i'
    else: raise

    # Data
    layers = []
    output_size = 0

    # Open the FIFO
    io = os.open(fifo_name, os.O_RDONLY)

    ### Read preliminary information
    # Number of layers
    num_layers = read_val(io, 'i')

    # Read layer information
    for i in xrange(num_layers):
        input_index = read_val(io, 'i')
        output_index = read_val(io, 'i')
        rows = read_val(io, 'i')
        columns = read_val(io, 'i')
        is_input = read_val(io, 'i')
        is_output = read_val(io, 'i')
        output_size += rows * columns
        layers.append(Layer(input_index, output_index,
            rows, columns,
            is_input != 0, is_output != 0,
            out_type))

    # Reading loop
    done = False
    while not done:
        try:
            # Read
            for layer in layers:
                if layer.is_output:
                    read_vals(io, fmt, layer.size, layer.output)
            # Print
            for layer in layers:
                layer.print_out()
        except OSError: break

if __name__ == "__main__":
    fifo_name = sys.argv[1]
    out_type = sys.argv[2]
    main(fifo_name, out_type)
    PyApp()
    gtk.main()

    print("Exiting!")
