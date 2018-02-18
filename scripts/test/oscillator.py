from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug

from os import path
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

def build_network(dim=64):
    plastic = "false"
    learning_rate = 0.01

    exc_exc_spread = 35
    exc_inh_spread = 35
    inh_exc_spread = 35
    inh_inh_spread = 35
    exc_inh_mask = 0

    exc_tau = 0.05
    inh_tau = 0.05

    exc_decay = 0.05
    inh_decay = 0.05

    # Noise Parameters
    exc_noise_strength = 1.0 / exc_tau
    exc_noise_rate = 10
    inh_noise_strength = 1.0 / inh_tau
    inh_noise_rate = 0
    exc_random = "true"
    inh_random = "false"

    # Weight parameters
    exc_exc_weight_init = "power law"
    exc_exc_exponent = 1.5
    exc_exc_base_weight = 0.1
    exc_exc_base_weight_min = 0.00011
    exc_exc_base_weight_max = 0.2
    exc_exc_fraction = 0.1

    exc_inh_weight_init = "power law"
    exc_inh_exponent = 1.5
    exc_inh_base_weight = 0.1
    exc_inh_base_weight_min = 0.00011
    exc_inh_base_weight_max = 0.2
    exc_inh_fraction = 0.1

    inh_exc_weight_init = "power law"
    inh_exc_exponent = 1.5
    inh_exc_base_weight = 0.1
    inh_exc_base_weight_min = 0.00011
    inh_exc_base_weight_max = 0.2
    inh_exc_fraction = 0.1

    inh_inh_weight_init = "power law"
    inh_inh_exponent = 1.5
    inh_inh_base_weight = 0.1
    inh_inh_base_weight_min = 0.00011
    inh_inh_base_weight_max = 0.2
    inh_inh_fraction = 0.1

    # Create main structure
    structure = {"name" : "oscillator", "type" : "parallel"}

    # Excitatory layer
    excitatory = {
        "name" : "exc",
        "neural model" : "oscillator",
        "rows" : dim,
        "columns" : dim,
        "tau" : exc_tau,
        "decay" : exc_decay
    }

    # Inhibitory layer
    inhibitory = {
        "name" : "inh",
        "neural model" : "oscillator",
        "rows" : dim,
        "columns" : dim,
        "tau" : inh_tau,
        "decay" : inh_decay
    }

    # Noise
    if exc_noise_rate > 0 and exc_noise_strength > 0:
        excitatory["noise config"] = {
            "type" : "poisson",
            "value" : exc_noise_strength,
            "rate" : exc_noise_rate,
            "random" : exc_random
        }
    if inh_noise_rate > 0 and inh_noise_strength > 0:
        inhibitory["noise config"] = {
            "type" : "poisson",
            "value" : inh_noise_strength,
            "rate" : inh_noise_rate,
            "random" : inh_random
        }


    # Add layers to structure
    structure["layers"] = [excitatory, inhibitory]

    exc_exc = {
        "from layer" : "exc",
        "to layer" : "exc",
        "name" : "exc exc matrix",
        "type" : "convergent",
        "arborized config" : {
            "field size" : exc_exc_spread,
            "wrap" : "true",
        },
        "opcode" : "add",
        "plastic" : plastic,
        "learning rate" : learning_rate,
        "max weight" : "0.5",
    }

    exc_inh = {
        "from layer" : "exc",
        "to layer" : "inh",
        "name" : "exc inh matrix",
        "type" : "convergent",
        "arborized config" : {
            "field size" : exc_inh_spread,
            "wrap" : "true",
        },
        "opcode" : "add",
        "plastic" : plastic,
        "learning rate" : learning_rate,
        "max weight" : "1.0",
    }

    inh_exc = {
        "from layer" : "inh",
        "to layer" : "exc",
        "name" : "inh exc matrix",
        "type" : "divergent",
        "arborized config" : {
            "field size" : inh_exc_spread,
            "wrap" : "true",
        },
        "opcode" : "sub",
        "plastic" : plastic,
        "learning rate" : learning_rate,
        "max weight" : "1.0",
    }

    inh_inh = {
        "from layer" : "inh",
        "to layer" : "inh",
        "name" : "inh inh matrix",
        "type" : "convergent",
        "arborized config" : {
            "field size" : inh_inh_spread,
            "wrap" : "true",
        },
        "opcode" : "sub",
        "plastic" : plastic,
        "learning rate" : learning_rate,
        "max weight" : "1.0",
    }

    # Exc Exc init
    if exc_exc_weight_init == "zero":
        exc_exc["weight config"] = {
                "type" : "flat",
                "weight" : 0.00011,
                "fraction" : exc_exc_fraction,
                "diagonal" : "false",
            }
    elif exc_exc_weight_init == "random":
        exc_exc["weight config"] = {
                "type" : "random",
                "min weight" : exc_exc_base_weight_min,
                "max weight" : exc_exc_base_weight_max,
                "fraction" : exc_exc_fraction,
                "diagonal" : "false",
            }
    elif exc_exc_weight_init == "flat":
        exc_exc["weight config"] = {
                "type" : "flat",
                "weight" : exc_exc_base_weight,
                "fraction" : exc_exc_fraction,
                "diagonal" : "false",
            }
    elif exc_exc_weight_init == "power law":
        exc_exc["weight config"] = {
                "type" : "power law",
                "exponent" : exc_exc_exponent,
                "min weight" : exc_exc_base_weight_min,
                "max weight" : exc_exc_base_weight_max,
                "fraction" : exc_exc_fraction,
                "diagonal" : "false",
            }

    # Exc Inh init
    if exc_inh_weight_init == "zero":
        exc_inh["weight config"] = {
                "type" : "flat",
                "weight" : 0.00011,
                "fraction" : exc_inh_fraction
            }
    elif exc_inh_weight_init == "random":
        exc_inh["weight config"] = {
                "type" : "random",
                "min weight" : exc_inh_base_weight_min,
                "max weight" : exc_inh_base_weight_max,
                "fraction" : exc_inh_fraction
            }
    elif exc_inh_weight_init == "flat":
        exc_inh["weight config"] = {
                "type" : "flat",
                "weight" : exc_inh_base_weight,
                "fraction" : exc_inh_fraction
            }
    elif exc_inh_weight_init == "power law":
        exc_inh["weight config"] = {
                "type" : "power law",
                "exponent" : exc_inh_exponent,
                "min weight" : exc_inh_base_weight_min,
                "max weight" : exc_inh_base_weight_max,
                "fraction" : exc_inh_fraction
            }

    # Inh Exc init
    if inh_exc_weight_init == "zero":
        inh_exc["weight config"] = {
                "type" : "flat",
                "weight" : 0.00011,
                "fraction" : inh_exc_fraction
            }
    elif inh_exc_weight_init == "random":
        inh_exc["weight config"] = {
                "type" : "random",
                "min weight" : inh_exc_base_weight_min,
                "max weight" : inh_exc_base_weight_max,
                "fraction" : inh_exc_fraction
            }
    elif inh_exc_weight_init == "flat":
        inh_exc["weight config"] = {
                "type" : "flat",
                "weight" : inh_exc_base_weight,
                "fraction" : inh_exc_fraction
            }
    elif inh_exc_weight_init == "power law":
        inh_exc["weight config"] = {
                "type" : "power law",
                "exponent" : inh_exc_exponent,
                "min weight" : inh_exc_base_weight_min,
                "max weight" : inh_exc_base_weight_max,
                "fraction" : inh_exc_fraction
            }

    # Inh Inh init
    if inh_inh_weight_init == "zero":
        inh_inh["weight config"] = {
                "type" : "flat",
                "weight" : 0.00011,
                "fraction" : inh_inh_fraction
            }
    elif inh_inh_weight_init == "random":
        inh_inh["weight config"] = {
                "type" : "random",
                "min weight" : inh_inh_base_weight_min,
                "max weight" : inh_inh_base_weight_max,
                "fraction" : inh_inh_fraction
            }
    elif inh_inh_weight_init == "flat":
        inh_inh["weight config"] = {
                "type" : "flat",
                "weight" : inh_inh_base_weight,
                "fraction" : inh_inh_fraction
            }
    elif inh_inh_weight_init == "power law":
        inh_inh["weight config"] = {
                "type" : "power law",
                "exponent" : inh_inh_exponent,
                "min weight" : inh_inh_base_weight_min,
                "max weight" : inh_inh_base_weight_max,
                "fraction" : inh_inh_fraction
            }

    # Create connections
    connections = [
        exc_exc,
        exc_inh,
        inh_exc,
        inh_inh
    ]

    # Create network
    return Network(
        {"structures" : [structure],
         "connections" : connections})

def build_environment(visualizer=False):
    modules = []
    if visualizer:
        modules = [
            {
                "type" : "visualizer",
                "colored" : "false",
                "layers" : [
                    { "structure" : "oscillator", "layer" : "exc" },
                    { "structure" : "oscillator", "layer" : "inh" },
                ]
            },
#            {
#                "type" : "visualizer",
#                "colored" : "true",
#
#                "decay" : "false",
#                "window" : 256,
#
#                #"decay" : "true",
#                #"window" : 1024,
#                #"bump" : 128,
#
#                "layers" : [
#                    { "structure" : "oscillator", "layer" : "exc" },
#                    { "structure" : "oscillator", "layer" : "inh" },
#                ]
#            }
        ]

    return Environment({"modules" : modules})

def main(infile=None, outfile=None, do_training=True, print_stats=True,
        visualizer=False, device=None, iterations=1000000):
    dim = 256

    network = build_network(dim)
    env = build_environment(visualizer)

    if infile is not None:
        if not path.exists(infile):
            print("Could not open state file: " + infile)
        else:
            print("Loading state from " + infile + " ...")
            network.load_state(infile)
            print("... done.")

    if device is None:
        device = get_gpus()[1]
    if do_training:
        report = network.run(env, {"multithreaded" : "true",
                                   "worker threads" : "1",
                                   "devices" : device,
                                   "iterations" : iterations,
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
    parser.add_argument('-s', action='store_true', default=False,
                        dest='stats',
                        help='print statistics')
    parser.add_argument('-v', action='store_true', default=False,
                        dest='visualizer',
                        help='run the visualizer')
    parser.add_argument('-host', action='store_true', default=False,
                        help='run on host CPU')
    parser.add_argument('-d', type=int, default=1,
                        help='run on device #')
    args = parser.parse_args()

    if args.host:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    main(args.i, args.o, args.train, args.stats, args.visualizer, device)
