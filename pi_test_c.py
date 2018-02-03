from syngen import Network, Environment, create_callback, FloatArray
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
        "columns" : cols,
        "noise config" : {
            "type" : "poisson",
            "value" : 0.5,
            "rate" : 10,
            "random" : "true"
        }}

    # Add layers to structure
    structure["layers"] = [field]

    # Create network
    return Network(
        {"structures" : [structure],
         "connections" : []})

def build_environment(rows, cols, visualizer=False):
    # Create environment modules
    modules = [
        {
            "type" : "socket",
            "rate" : 1,
            "layers" : [
                {
                    "structure" : "pi_test",
                    "layer" : "field",
                    "input" : "true",
                }
            ]
        },
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
    rows = 100
    cols = 100

    network = build_network(rows, cols)
    env = build_environment(rows, cols, visualizer)

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
        report = network.run(env, {"multithreaded" : "true",
                                   "worker threads" : "2",
                                   "devices" : device,
                                   "iterations" : iterations,
                                   "refresh rate" : rate,
                                   "verbose" : "true"})
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
