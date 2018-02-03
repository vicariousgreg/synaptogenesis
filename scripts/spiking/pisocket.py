import socket
import argparse
from struct import pack, unpack
#from picam import PiCam

class PiSocket:
    def get_ping(self):
        while not self.conn.recv(1): pass

    def send_ping(self):
        self.conn.sendall(' ')

    def get_integer(self):
        mesg = ''
        rec = 0
        while rec < 16:
            mesg += self.conn.recv(16 - rec)
            rec += len(mesg)
        return int(mesg)

    def send_integer(self, i):
        self.conn.sendall("%016d" % i)

    def get_data(self, size, count, buf):
        view = memoryview(buf)
        num_bytes = size * count
        while num_bytes:
            nbytes = self.conn.recv_into(view, num_bytes)
            view = view[nbytes:]
            num_bytes -= nbytes

    def send_data(self, mesg, ty="f"):
        self.conn.sendall(bytearray(pack(ty * len(mesg), *mesg)))

class PiServer(PiSocket):
    def __init__(self, TCP_IP='192.168.0.180', TCP_PORT=11111):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.bind((TCP_IP, TCP_PORT))
        self.s.listen(1)
        conn, addr = self.s.accept()
        self.conn = conn

    def close(self):
        self.s.close()

    def __del__(self):
        self.close()

class PiClient(PiSocket):
    def __init__(self, TCP_IP='192.168.0.180', TCP_PORT=12397):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((TCP_IP, TCP_PORT))

    def close(self):
        self.conn.close()

    def __del__(self):
        self.close()

def gen_mesg(message_size):
    return [uniform(0.0, 1.0) for _ in xrange(message_size)]

#def get_camera_mesg(cam):
#    return cam.capture_greyscale()

if __name__ == "__main__":
    import time
    from random import uniform

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store_true', default=False,
                        dest='server',
                        help='run server process (client is default)')
    parser.add_argument('-v', action='store_true', default=False,
                        dest='verbose',
                        help='print as messages are passed, and time stats')
    parser.add_argument('-get', action='store_true', default=False,
                        dest='get',
                        help='gets messages')
    parser.add_argument('-send', action='store_true', default=False,
                        dest='send',
                        help='send messages')
    parser.add_argument('-n', type=int, default=1000,
                        help='number of messages to exchange')
    parser.add_argument('-m', type=int, default=10000,
                        help='size of messages to exchange')
    parser.add_argument('-p', type=int, default=12397,
                        help='port to use')
    args = parser.parse_args()

    TCP_IP = '192.168.0.180'
    TCP_PORT = 12397

    message_count = args.n
    message_size = args.m

    if not args.get and not args.send:
        args.get = args.server
        args.send = not args.server

#    if args.send: cam = PiCam()
    sock = PiServer(TCP_PORT=args.p) if args.server else PiClient(TCP_PORT=args.p)

    try:
        if args.get:
            buf = bytearray(4 * message_size)

        start = time.time()

        # Send first message from client
        if not args.server and args.send:
            sock.get_ping()
            mesg = gen_mesg(message_size)
#            mesg = get_camera_mesg(cam)
            sock.send_data(mesg)
            if args.verbose: print("Sent")

        # Exchange
        for i in xrange(message_count-1):
            if args.get:
                sock.send_ping() if args.server else sock.get_ping()

                sock.get_data(4, message_size, buf)
                vals = unpack("f" * message_size, buf)
                if args.verbose:
                    print("Got")
                    print(min(vals), max(vals), sum(vals) / message_size)
            if args.send:
                sock.send_ping() if args.server else sock.get_ping()

                mesg = gen_mesg(message_size)
#                mesg = get_camera_mesg(cam)
                sock.send_data(mesg)
                if args.verbose: print("Sent")

        # Get last message from client
        if args.server and args.get:
            sock.send_ping()
            sock.get_data(4, message_size, buf)
            if args.verbose: print("Got")

        if args.verbose: print("Elapsed time: %f" % (time.time() - start))
    except Exception as e: print(e)
    sock.close()
