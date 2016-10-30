import os

class Layer:
    def __init__(self, name, rows=100, cols=100, params="none"):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.params = params

class Model:
    def __init__(self, path):
        self.layers = []
        self.lines = []
        for line in [l.strip() for l in open(path).readlines() if len(l.strip()) > 0]:
            #print(line)
            self.lines.append(line)

    def send(self, fifo_name):
        fifo = os.open(fifo_name, os.O_WRONLY)
        for line in self.lines:
            os.write(fifo, line + "\n")
