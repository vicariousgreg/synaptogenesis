import os
import struct

class Layer:
    def __init__(self, index, rows, columns):
        self.output = [0] * (rows * columns)
        self.index = index
        self.rows = rows
        self.columns = columns
        self.size = rows * columns
        print("Layer: (index=%d, rows=%d, cols=%d)" % (index, rows, columns))

def read_int(f):
    line = os.read(f, 4)
    if len(line) > 0:
        return struct.unpack('i', line)[0]
    else: raise OSError

def main():
    # Data
    layers = []
    output_size = 0

    # Open the FIFO
    fifo = "/tmp/pcnn_fifo"
    io = os.open(fifo, os.O_RDONLY)

    ### Read preliminary information
    # Number of layers
    num_layers = read_int(io)

    # Read layer information
    for i in xrange(num_layers):
        index = read_int(io)
        rows = read_int(io)
        columns = read_int(io)
        output_size += rows * columns
        layers.append(Layer(index, rows, columns))

    done = False
    while not done:
        for layer in layers:
            line = os.read(io, layer.size * 4)
            if len(line) > 0:
                for i in xrange(layer.size):
                    layer.output[i] = struct.unpack_from('i', line, i * 4)[0]
                #print(len([x for x in layer.output if x > 0]), layer.size)
            else: done = True

if __name__ == "__main__":
    main()
    print("Exiting!")
