from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug
from pisocket import PiServer
from ctypes import cast, POINTER, c_byte

from os import path
import sys
import argparse


def build_network(rows=480, cols=720):
    # Create main structure (parallel engine)
    structure = {"name" : "pi_test", "type" : "parallel"}

    # Add layers
    field = {
        "name" : "field",
        "neural model" : "relay",
        "rows" : rows,
        "columns" : cols}

    # Add layers to structure
    structure["layers"] = [field]

    # Create network
    return Network(
        {"structures" : [structure],
         "connections" : []})

def build_environment(sensory_socket, motor_socket, rows, cols, visualizer=False):
    buf = bytearray(rows * cols * 4)

    def sensory_callback(ID, length, ptr):
        sensory_socket.send_ping()
        sensory_socket.get_data(4, length, buf)

        c_buf = cast(ptr, POINTER(c_byte))
        for i,x in enumerate(buf):
            c_buf[i] = x

    def motor_callback(ID, length, ptr):
        motor_socket.send_ping()
        arr = FloatArray(length, ptr)
        motor_socket.send_data(arr.to_list())

    create_io_callback("sensory", sensory_callback)
    create_io_callback("motor", motor_callback)

    # Create environment modules
    modules = [
        {
            "type" : "callback",
            "rate" : 1,
            "layers" : [
                {
                    "structure" : "pi_test",
                    "layer" : "field",
                    "input" : True,
                    "function" : "sensory",
                    "id" : 0
                }
            ]
        },
        {
            "type" : "callback",
            "rate" : 1,
            "layers" : [
                {
                    "structure" : "pi_test",
                    "layer" : "field",
                    "output" : True,
                    "function" : "motor",
                    "id" : 0
                }
            ]
        }
    ]

    if visualizer:
        modules.append({
            "type" : "visualizer",
            "layers" : [
                { "structure" : "pi_test", "layer" : "field" }
            ]
        })

    return Environment({"modules" : modules})

def main(infile=None, outfile=None, do_training=True,
        visualizer=False, device=None, rate=0, iterations=1000):
    rows = 480
    cols = 720

    sensory_socket = PiServer(TCP_PORT=11111)
    motor_socket = PiServer(TCP_PORT=11112)
    network = build_network(rows, cols)
    env = build_environment(sensory_socket, motor_socket, rows, cols, visualizer)

    if infile is not None:
        if not path.exists(infile):
            print("Could not open state file: " + infile)
        else:
            print("Loading state from " + infile + " ...")
            network.load_state(infile)
            print("... done.")

    if device is None:
        device = gpus[len(gpus)-1] if len(gpus) > 0 else get_cpu()
    if do_training:
        report = network.run(env, {"multithreaded" : True,
                                   "worker threads" : "2",
                                   "devices" : device,
                                   "iterations" : iterations,
                                   "refresh rate" : rate,
                                   "verbose" : True})
        if report is None:
            print("Engine failure.  Exiting...")
            return
        print(report)

        if outfile is not None:
            print("Saving state to " + outfile + " ...")
            network.save_state(outfile)
            print("... done.")

    # Delete the objects
    del network
    del env
    sensory_socket.close()
    motor_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str,
                        help='source state file')
    parser.add_argument('-o', type=str,
                        help='destination state file')
    parser.add_argument('-t', action='store_true', default=False,
                        dest='train',
                        help='run training')
    parser.add_argument('-v', action='store_true', default=False,
                        dest='visualizer',
                        help='run the visualizer')
    parser.add_argument('-host', action='store_true', default=False,
                        help='run on host CPU')
    parser.add_argument('-d', type=int, default=1,
                        help='run on device #')
    parser.add_argument('-r', type=int, default=0,
                        help='refresh rate')
    args = parser.parse_args()

    if args.host or len(get_gpus()) == 0:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    main(args.i, args.o, args.train, args.visualizer, device, args.r)
