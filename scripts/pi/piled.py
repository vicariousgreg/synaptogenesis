from Adafruit_LED_Backpack import Matrix8x8

class PiLED:
    def __init__(self):
        self.disp = Matrix8x8.Matrix8x8()
        self.disp.begin()
        self.coordinates = [
            (row, col, row*8+col)
                for row in xrange(8)
                    for col in xrange(8)]

    def clear(self):
        self.disp.clear()
        self.disp.write_display()

    def display(self, arr):
        self.disp.clear()
        for row,col,i in self.coordinates:
            self.disp.set_pixel(row, col, arr[i] > 0.1)
        self.disp.write_display()

if __name__ == "__main__":
    from time import time
    from random import random

    led = PiLED()
    start = time()
    for x in xrange(100):
        led.display([0.15 * random() for _ in xrange(64)])
    print(time() - start)
    led.clear()
