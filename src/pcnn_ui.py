import os
import struct
import sys

class Layer:
    def __init__(self, index, rows, columns):
        self.output = [0] * (rows * columns)
        self.index = index
        self.rows = rows
        self.columns = columns
        self.size = rows * columns
        print("Layer: (index=%d, rows=%d, cols=%d)" % (index, rows, columns))

    def print_out(self):
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

def read_int(f):
    line = os.read(f, 4)
    if len(line) > 0:
        return struct.unpack('i', line)[0]
    else: raise OSError

def main(fifo_name):
    # Data
    layers = []
    output_size = 0

    # Open the FIFO
    io = os.open(fifo_name, os.O_RDONLY)

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

    # Reading loop
    done = False
    while not done:
        for layer in layers:
            line = os.read(io, layer.size * 4)
            if len(line) > 0:
                for i in xrange(layer.size):
                    layer.output[i] = struct.unpack_from('i', line, i * 4)[0]
                #print(len([x for x in layer.output if x > 0]), layer.size)
            else: done = True
        for layer in layers:
            layer.print_out()

if __name__ == "__main__":
    fifo_name = sys.argv[1]
    main(fifo_name)
    print("Exiting!")
