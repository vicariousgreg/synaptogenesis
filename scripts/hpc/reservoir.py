from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug

from random import random
from os import path
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

def build_rc(name, device, fraction, input_size, res_dim, output_size):
    # Create main structure
    structure = {"name" : name, "type" : "parallel"}

    # Input layer
    input_layer = {
        "name" : "input_layer",
        "neural model" : "sine generator",
        "rows" : 1,
        "columns" : input_size,
        "frequency" : 100,
        "init config" : {
            "type" : "uniform",
            "min" : 0.0,
            "max" : 1.0,
        },
        "device" : device,
    }
    reservoir = {
        "name" : "reservoir",
        "neural model" : "nvm_tanh",
        "rows" : res_dim,
        "columns" : res_dim,
        "device" : device,
    }
    output_layer = {
        "name" : "output_layer",
        "neural model" : "nvm_tanh",
        "rows" : 1,
        "columns" : output_size,
        "device" : device,
    }


    # Add layers to structure
    structure["layers"] = [ input_layer, reservoir, output_layer ]

    r_i = {
        "name" : "r_i",
        "from structure" : name,
        "from layer" : "input_layer",
        "to structure" : name,
        "to layer" : "reservoir",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "sparse" : fraction < 0.5,
        "weight config" : {
            "type" : "random",
            "min weight" : -1.0,
            "max weight" : 1.0,
            "fraction" : fraction
        },
    }

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
        "weight config" : {
            "type" : "random",
            "min weight" : -0.1 / (res_dim * res_dim),
            "max weight" : 0.1 / (res_dim * res_dim),
            "fraction" : fraction
        },
    }

    o_r = {
        "name" : "o_r",
        "from structure" : name,
        "from layer" : "reservoir",
        "to structure" : name,
        "to layer" : "output_layer",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "sparse" : fraction < 0.5,
        "weight config" : {
            "type" : "random",
            "min weight" : -10.0 / (res_dim * res_dim),
            "max weight" : 10.0 / (res_dim * res_dim),
            "fraction" : fraction
        },
    }

    # Create connections
    connections = [ r_i, r_r, o_r ]

    # Create network
    return structure, connections

def link_rc(structures, res_dim, macro_fraction=0.1, micro_fraction=0.1):
    conn = []

    for s1 in structures:
        for s2 in structures:
            if s1 != s2 and random() < macro_fraction:
                conn.append(
                    {
                        "name" : "%s>%s" % (s1["name"], s2["name"]),
                        "from structure" : s1["name"],
                        "from layer" : "output_layer",
                        "to structure" : s2["name"],
                        "to layer" : "input_layer",
                        "type" : "fully connected",
                        "opcode" : "add",
                        "plastic" : False,
                        "sparse" : micro_fraction < 0.5,
                        "weight config" : {
                            "type" : "random",
                            #"min weight" : -0.1 / (res_dim * res_dim),
                            #"max weight" : 0.1 / (res_dim * res_dim),
                            "min weight" : -1.0,
                            "max weight" : 1.0,
                            "fraction" : micro_fraction
                        },
                    })
    return conn

def build_environment(num_rcs, visualizer=False):
    modules = []
    if visualizer:
        modules = [
            {
                "type" : "visualizer",
                "colored" : False,
                "layers" : [
                    #{ "structure" : "rc", "layer" : "input_layer" },
                    { "structure" : "rc%05d" % i, "layer" : "reservoir" }
                        for i in range(num_rcs)
                    #{ "structure" : "rc", "layer" : "output_layer" },
                ]
            },
#            {
#                "type" : "periodic_input",
#                "min" : -0.5,
#                "max" : 0.5,
#                "fraction" : 0.5,
#                "random" : True,
#                "layers" : [
#                    {
#                        "structure" : "rc",
#                        "layer" : "input_layer"
#                    }
#                ]
#            }
        ]

    return Environment({"modules" : modules})

def main(infile=None, outfile=None,
        visualizer=False, refresh_rate=0, device=None,
        iterations=1000000, worker_threads=4,
        engine_multithreading=True):

    num_rcs = 32
    input_size = 32
    res_dim = 64
    output_size = 32
    fraction = 1.0

    if type(device) is list:
        devices = [device[i % len(device)] for i in range(num_rcs)]
    else:
        devices = [device for _ in range(num_rcs)]

    structures = []
    connections = []
    for i in range(num_rcs):
        name = "rc%05d" % i
        s, c = build_rc(name, devices[i], fraction,
            input_size, res_dim, output_size)
        structures.append(s)
        connections += c

    connections += link_rc(structures, res_dim, 0.1, 1.0)

    network = Network(
        {"structures" : structures,
         "connections" : connections})

    env = build_environment(num_rcs, visualizer)

    if infile is not None:
        if not path.exists(infile):
            print("Could not open state file: " + infile)
        else:
            print("Loading state from " + infile + " ...")
            network.load_state(infile)
            print("... done.")

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

    #mat = network.get_weight_matrix("r_i").to_list()
    #print(mat)

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
    parser.add_argument('-v', action='store_true', default=False,
                        dest='visualizer',
                        help='run the visualizer')
    parser.add_argument('-host', action='store_true', default=False,
                        help='run on host CPU')
    parser.add_argument('-d', type=int, default=1,
                        help='run on device #')
    parser.add_argument('-r', type=int, default=0,
                        help='refresh rate')
    parser.add_argument('-it', type=int, default=1000000,
                        help='iterations')
    parser.add_argument('-w', type=int, default=4,
                        help='worker threads')
    args = parser.parse_args()

    if args.host or len(get_gpus()) == 0:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]
    #device = get_gpus()

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    engine_multithreading = False

    main(args.i, args.o, args.visualizer, args.r, device, args.it,
        args.w, engine_multithreading)
