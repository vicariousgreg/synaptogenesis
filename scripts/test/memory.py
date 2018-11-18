from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug

from random import random
from os import path
import sys
import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

def build_rc(name, device, fraction, res_dim):
    # Create main structure
    structure = {"name" : name, "type" : "parallel"}

    reservoir = {
        "name" : "reservoir",
        "neural model" : "nvm_tanh",
        "rows" : res_dim,
        "columns" : res_dim,
        "device" : device,
    }


    # Add layers to structure
    structure["layers"] = [ reservoir ]

    r_r = {
        "name" : "r_r",
        "from structure" : name,
        "from layer" : "reservoir",
        "to structure" : name,
        "to layer" : "reservoir",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "sparse" : fraction < 0.5,
    }

    # Create connections
    connections = [ r_r ]

    # Create network
    return structure, connections

def main(infile=None, outfile=None,
        visualizer=False, refresh_rate=0, device=None,
        iterations=1000000, worker_threads=4,
        engine_multithreading=True):

    num_rcs = 8
    res_dim = 64
    fraction = 1.0

    if type(device) is list:
        devices = [device[i % len(device)] for i in range(num_rcs)]
    else:
        devices = [device for _ in range(num_rcs)]

    structures = []
    connections = []
    for i in range(num_rcs):
        name = "rc%05d" % i
        s, c = build_rc(name, devices[i], fraction, res_dim)
        structures.append(s)
        connections += c

    network = Network(
        {"structures" : structures,
         "connections" : connections})

    env = Environment({"modules" : []})

    if device is None:
        device = get_gpus()[1]

    train_args = {"multithreaded" : engine_multithreading,
                  "worker threads" : worker_threads,
                  "devices" : device,
                  "iterations" : iterations,
                  "verbose" : True}
    if refresh_rate > 0:
        train_args["refresh rate"] = refresh_rate;
    report = network.run(env, train_args)

    if report is None:
        print("Engine failure.  Exiting...")
        return
    print(report)

    if outfile is not None:
        print("Saving state to " + outfile + " ...")
        network.save_state(outfile)
        print("... done.")

    # Delete the objects
    print("*" * 80)
    print("Deleting objects...")
    network.free()
    env.free()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str,
                        help='source state file')
    parser.add_argument('-o', type=str,
                        help='destination state file')
    parser.add_argument('-v', action='store_true', default=False,
                        dest='visualizer',
                        help='run the visualizer')
    parser.add_argument('-host', action='store_true', default=False,
                        help='run on host CPU')
    parser.add_argument('-gpus', action='store_true', default=False,
                        help='run on available GPUs')
    parser.add_argument('-d', type=int, default=1,
                        help='run on device #')
    parser.add_argument('-r', type=float, default=0.0,
                        help='refresh rate')
    parser.add_argument('-it', type=int, default=1000000,
                        help='iterations')
    parser.add_argument('-w', type=int, default=0,
                        help='worker threads')
    parser.add_argument('-e', action='store_true', default=False,
                        help='engine multithreading')
    args = parser.parse_args()

    if args.host and args.gpus:
        device = get_gpus() + [get_cpu()]
    elif args.host or len(get_gpus()) == 0:
        device = get_cpu()
    elif args.gpus:
        device = get_gpus()
    else:
        device = get_gpus()[args.d]

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    for _ in range(100):
        main(args.i, args.o, args.visualizer, args.r, device, args.it, args.w, args.e)
        raw_input("Press any key to continue...")
