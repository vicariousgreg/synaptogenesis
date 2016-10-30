import sys

from model import Model

if __name__ == "__main__":
    fifo_name = sys.argv[1]
    path = sys.argv[2]

    model = Model(path)
    model.send(fifo_name)
