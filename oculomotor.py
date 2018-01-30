from syngen import Network, Environment
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug
from math import exp

from os import path
import sys
import argparse

def gaussian_weight_string(rows, cols, peak, sig, norm=False):
    inv_sqrt_2pi = 0.3989422804014327

    row_center = rows / 2
    col_center = cols / 2
    peak_coeff = peak * (1.0 if norm else inv_sqrt_2pi / sig)
    gauss = []

    for row in xrange(rows):
        for col in xrange(cols):
            dist = ((row - row_center) ** 2) + ((col - col_center) ** 2) ** 0.5
            a = dist / sig
            gauss.append("%.6f" % (peak_coeff * exp(-0.5 * a * a)))
    return " ".join(gauss)

def build_network(dim=200):
    # Create main structure (parallel engine)
    structure = {"name" : "oculomotor", "type" : "parallel"}

    exc_tau = 0.1
    inh_tau = 0.1

    exc_decay = 0.05
    inh_decay = 0.05

    exc_noise_strength = 0.5 / exc_tau
    exc_noise_rate = 10
    inh_noise_strength = 0.5 / inh_tau
    inh_noise_rate = 0
    exc_noise_random = "false"
    inh_noise_random = "false"

    receptive_field = 15

    exc_exc_rf = 7
    exc_inh_rf = 31
    inh_exc_rf = 21
    inh_inh_rf = 31

    mask_rf = 7

    exc_exc_fraction = 0.5
    exc_inh_fraction = 0.5
    inh_exc_fraction = 0.5
    inh_inh_fraction = 0.5

    exc_exc_mean = 0.2
    exc_inh_mean = 0.1
    inh_exc_mean = 0.1
    inh_inh_mean = 0.1

    exc_exc_std_dev = 0.025
    exc_inh_std_dev = 0.025
    inh_exc_std_dev = 0.025
    inh_inh_std_dev = 0.025

    # Add layers
    vision_layer = {
        "name" : "vision",
        "neural model" : "oscillator",
        "rows" : dim,
        "columns" : dim}
    sc_exc = {
        "name" : "sc_exc",
        "neural model" : "oscillator",
        "rows" : dim,
        "columns" : dim,
        "tau" : exc_tau,
        "decay" : exc_decay,
        "noise config" : {
            "type" : "poisson",
            "value" : exc_noise_strength,
            "rate" : exc_noise_rate,
            "random" : exc_noise_random
        }}
    sc_inh = {
        "name" : "sc_inh",
        "neural model" : "oscillator",
        "rows" : dim,
        "columns" : dim,
        "tau" : inh_tau,
        "decay" : inh_decay,
        "noise config" : {
            "type" : "poisson",
            "value" : inh_noise_strength,
            "rate" : inh_noise_rate,
            "random" : inh_noise_random
        }}

    # Add layers to structure
    structure["layers"] = [vision_layer, sc_exc, sc_inh]

    # Create connections
    connections = [
        {
            "from layer" : "vision",
            "to layer" : "sc_exc",
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.1,
    #            "weight string" : gaussian_weight_string(
    #                receptive_field, receptive_field, 3, 3, False)
            },
            "arborized config" : {
                "field size" : receptive_field,
                "stride" : 1,
                "wrap" : "false"
            }
        },
        {
            "from layer" : "sc_exc",
            "to layer" : "sc_exc",
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "gaussian",
                "mean" : exc_exc_mean,
                "std dev" : exc_exc_std_dev,
                "fraction" : exc_exc_fraction,
                "circular mask" : [ { } ]
            },
            "arborized config" : {
                "field size" : exc_exc_rf,
                "stride" : 1,
            }
        },
        {
            "from layer" : "sc_exc",
            "to layer" : "sc_inh",
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "gaussian",
                "mean" : exc_inh_mean,
                "std dev" : exc_inh_std_dev,
                "fraction" : exc_inh_fraction,
                "circular mask" : [
                    {
                        "diameter" : mask_rf,
                        "invert" : "true"
                    },
                    { }
                ]
            },
            "arborized config" : {
                "field size" : exc_inh_rf,
                "stride" : 1,
            }
        },
        {
            "from layer" : "sc_inh",
            "to layer" : "sc_exc",
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "sub",
            "plastic" : "false",
            "weight config" : {
                "type" : "gaussian",
                "mean" : inh_exc_mean,
                "std dev" : inh_exc_std_dev,
                "fraction" : inh_exc_fraction,
                "circular mask" : [ { } ]
            },
            "arborized config" : {
                "field size" : inh_exc_rf,
                "stride" : 1,
            }
        },
#        {
#            "from layer" : "sc_inh",
#            "to layer" : "sc_inh",
#            "type" : "convergent",
#            "convolutional" : "true",
#            "opcode" : "sub",
#            "plastic" : "false",
#                "weight config" : {
#                "type" : "gaussian",
#                "mean" : inh_inh_mean,
#                "std dev" : inh_inh_std_dev,
#                "fraction" : inh_inh_fraction,
#                "circular mask" : [ { } ]
#            },
#            "arborized config" : {
#                "field size" : inh_inh_rf,
#                "stride" : 1,
#            }
#        },
    ]
    # Create network
    return Network(
        {"structures" : [structure],
         "connections" : connections})

def build_environment(visualizer=False):
    # Create environment modules
    modules = [
        {
            "type" : "visualizer",
            "layers" : [
                { "structure" : "oculomotor", "layer" : "vision" },
                { "structure" : "oculomotor", "layer" : "sc_exc" },
                { "structure" : "oculomotor", "layer" : "sc_inh" }
            ]
        },
        {
            "type" : "gaussian_random_input",
            "rate" : "200",
            "std dev" : "5",
            "value" : "0.25",
            "normalize" : "true",
            "peaks" : "3",
            "random" : "true",
            "layers" : [
                {
                    "structure" : "oculomotor",
                    "layer" : "vision"
                }
            ]
        }
    ]

    return Environment({"modules" : modules})

def main(infile=None, outfile=None, do_training=True,
        visualizer=False, device=None, rate=0, iterations=1000000):
    dim = 200

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
        device = gpus[len(gpus)-1] if len(gpus) > 0 else get_cpu()
    if do_training:
        report = network.run(env, {"multithreaded" : "true",
                                   "worker threads" : "4",
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

    if args.host:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    main(args.i, args.o, args.train, args.visualizer, device, args.r)
