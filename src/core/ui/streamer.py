import os
import struct

import gtk
import gobject

from layer import Layer, convert_spikes

class Streamer:
    def __init__(self, fifo_name, out_type):
        # Open FIFO
        self.fifo = os.open(fifo_name, os.O_RDONLY)

        # Determine output format
        self.out_type = out_type
        if out_type == "float":
            self.fmt = 'f'
        elif out_type == "int":
            self.fmt = 'i'
        elif out_type == "bit":
            self.fmt = 'i'
        else: raise

    def read_output(self, rows, columns, dest):
        line = os.read(self.fifo, rows * columns * 4)
        if len(line) == 0: return False
        else:
            for i in xrange(rows):
                for j in xrange(columns):
                    index = i * columns + j
                    val = convert_spikes(struct.unpack_from(self.fmt, line, index * 4)[0])
                    dest[i][j][0] = val
                    dest[i][j][1] = val
                    dest[i][j][2] = val
            return True

    def read_val(self, fmt):
        line = os.read(self.fifo, 4)
        if len(line) > 0:
            return struct.unpack(fmt, line)[0]
        else: raise OSError

    def read_vals(self, fmt, count, dest):
        line = os.read(self.fifo, count * 4)
        if len(line) > 0:
            for i in xrange(count):
                val = struct.unpack_from(fmt, line, i * 4)[0]
                dest[i][0] = val
                dest[i][1] = val
                dest[i][2] = val
        else: raise OSError

    def read_layers(self):
        # Data
        layers = []
        output_size = 0

        ### Read preliminary information
        # Number of layers
        num_layers = self.read_val('i')

        # Read layer information
        for i in xrange(num_layers):
            input_index = self.read_val('i')
            output_index = self.read_val('i')
            rows = self.read_val('i')
            columns = self.read_val('i')
            is_input = self.read_val('i')
            is_output = self.read_val('i')
            output_size += rows * columns
            layers.append(Layer(input_index, output_index,
                rows, columns,
                is_input != 0, is_output != 0,
                self.out_type))
        return layers

    def read_loop(self, layers, read_sender, kill_sender):
        # Reading loop
        done = False
        while not done:
            try:
                # Read
                for layer in layers:
                    if layer.is_output:
                        if not (self.read_output(
                            layer.rows, layer.columns, layer.pixbuf.get_pixels_array())):
                            done = True
                # Print
                #for layer in layers:
                #    layer.print_out()
            except OSError: break
            read_sender.send()
        kill_sender.send()
