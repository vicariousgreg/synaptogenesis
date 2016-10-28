import threading
import os
import struct
import sys

import gtk
import gobject

from layer import Layer
from ui_app import PyApp

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

def read_layers(io):
    # Data
    layers = []
    output_size = 0

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
    return layers

def read_loop(layers, io, fmt):
    # Reading loop
    while True:
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

    # Determine output format
    if out_type == "float":
        fmt = 'f'
    elif out_type == "int":
        fmt = 'i'
    elif out_type == "bit":
        fmt = 'i'
    else: raise


    # Open the FIFO
    io = os.open(fifo_name, os.O_RDONLY)

    # Read layer information from FIFO
    layers = read_layers(io)

    # Launch read thread
    thread = threading.Thread(target=read_loop, args=(layers, io, fmt))
    thread.daemon = True
    thread.start()

    gobject.threads_init()
    PyApp()

    print("Exiting!")
