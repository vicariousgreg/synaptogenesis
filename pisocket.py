import socket
import argparse
from struct import pack

class PiSocket:
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
    def __init__(self, TCP_IP='192.168.0.180', TCP_PORT=12397):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
    parser.add_argument('-n', type=int, default=1000,
                        help='number of messages to exchange')
    parser.add_argument('-m', type=int, default=10000,
                        help='size of messages to exchange')
    args = parser.parse_args()

    TCP_IP = '192.168.0.180'
    TCP_PORT = 12397

    message_count = args.n
    message_size = args.m

    sock = PiServer() if args.server else PiClient()

    try:
        start = time.time()
        buf = bytearray(4 * message_size)

        # Send first message from client
        if not args.server:
            mesg = [uniform(0.0, 1.0) for _ in xrange(message_size)]
            sock.send_data(mesg)
            if args.verbose: print("Sent")

        # Exchange
        for i in xrange(message_count-1):
            sock.get_data(4, message_size, buf)
            if args.verbose: print("Got")
            mesg = [uniform(0.0, 1.0) for _ in xrange(message_size)]
            sock.send_data(mesg)
            if args.verbose: print("Sent")

        # Get last message from client
        if args.server:
            sock.get_data(4, message_size, buf)
            if args.verbose: print("Got")

        if args.verbose: print("Elapsed time: %f" % (time.time() - start))
    except Exception as e: print(e)
    sock.close()
