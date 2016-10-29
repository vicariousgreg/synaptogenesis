from gtk.gdk import Pixbuf, COLORSPACE_RGB

class Layer:
    def __init__(self, input_index, output_index,
            rows, columns, is_input, is_output, out_type):
        self.pixbuf = Pixbuf(COLORSPACE_RGB, False, 8, columns, rows)
        pix = self.pixbuf.get_pixels_array()
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
        pixels = self.pixbuf.get_pixels_array()
        for i in xrange(self.rows):
            for j in xrange(self.columns):
                val = pixels[i][j][0]

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
        pixels = self.pixbuf.get_pixels_array()
        for i in xrange(self.rows):
            for j in xrange(self.columns):
                val = pixels[i][j][0]
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
