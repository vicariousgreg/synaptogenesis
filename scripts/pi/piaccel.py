import smbus
from math import sqrt, degrees, atan2

def dist(a,b):
    return sqrt((a*a)+(b*b))

class PiAccel:
    def __init__(self):
        self.power_mgmt_1 = 0x6b
        self.power_mgmt_2 = 0x6c
        self.bus = smbus.SMBus(1)
        self.address = 0x68
        self.bus.write_byte_data(self.address, self.power_mgmt_1, 0)

    def read_word(self, reg):
        val = ((
            self.bus.read_byte_data(self.address, reg) << 8) +
            self.bus.read_byte_data(self.address, reg+1))
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

    def get_gyro(self):
        return (self.read_word(0x43) / 65535.0,
            self.read_word(0x45) / 65535.0,
            self.read_word(0x47) / 65535.0)

    def get_accel(self):
        return (self.read_word(0x3b) / 65535.0,
            self.read_word(0x3d) / 65535.0,
            self.read_word(0x3f) / 65535.0)

    def get_rotation(self):
        x,y,z = (self.read_word(0x3b) / 16384.0,
            self.read_word(0x3d) / 16384.0,
            self.read_word(0x3f) / 16384.0)
        return (degrees(atan2(y, dist(x,z))),
            -degrees(atan2(x, dist(y,z))))

if __name__ == "__main__":
    from time import time

    accel = PiAccel()
    start = time()
    for x in xrange(10):
        print(accel.get_gyro())
        print(accel.get_accel())
        print(accel.get_rotation())
        print("")
    print(time() - start)
