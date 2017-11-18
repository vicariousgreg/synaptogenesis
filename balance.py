from syngen import Network, Environment, create_callback, FloatArray
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug

from os import path
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

def main(infile=None, outfile=None, silent=False, visualizer=False):
    leaky = False
    plastic = "true"
    learning_rate = 0.005
    base_weight = 0.001

    exc_noise_strength = 10.0
    exc_noise_rate = 30
    inh_noise_strength = 10.0
    inh_noise_rate = 30

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    # Create main structure (feedforward engine)
    structure = {"name" : "snn", "type" : "parallel"}

    dim = 64
    exc_exc_spread = 25
    exc_inh_spread = 15
    inh_exc_spread = 15

    # Excitatory layer
    excitatory = {
        "name" : "exc",
        "neural model" : ("leaky_izhikevich" if leaky else "izhikevich"),
        "rows" : dim,
        "columns" : dim,
        "noise config" : {
            "type" : "poisson",
            "value" : exc_noise_strength,
            "rate" : exc_noise_rate
        },
        "neuron spacing" : "0.1",
        "init" : "regular"
        #"init" : "random positive"
    }

    # Inhibitory layer
    inhibitory = {
        "name" : "inh",
        "neural model" : ("leaky_izhikevich" if leaky else "izhikevich"),
        "rows" : dim/2,
        "columns" : dim/2,
        "noise config" : {
            "type" : "poisson",
            "value" : 10.0,
            "value" : inh_noise_strength,
            "rate" : inh_noise_rate
        },
        "neuron spacing" : "0.2",
        "init" : "fast"
        #"init" : "random negative"
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
        "weight config" : {
    #        "type" : "power law",
    #        "exponent" : "1.5",
            "type" : "flat",
            "weight" : base_weight,
            "fraction" : "1.0"
        },
    }

    exc_inh = {
        "from layer" : "exc",
        "to layer" : "inh",
        "name" : "exc inh matrix",
        "type" : "convergent",
        "arborized config" : {
            "field size" : exc_inh_spread,
            "stride" : 2,
            "wrap" : "true",
        },
        "opcode" : "add",
        "plastic" : plastic,
        "learning rate" : learning_rate,
        "max weight" : "0.5",
        "weight config" : {
    #        "type" : "power law",
    #        "exponent" : "1.5",
            "type" : "flat",
            "weight" : base_weight,
            "fraction" : "1.0"
        },
    }
    inh_exc = {
        "from layer" : "inh",
        "to layer" : "exc",
        "name" : "inh exc matrix",
        "type" : "divergent",
        "arborized config" : {
            "field size" : inh_exc_spread,
            "stride" : 2,
            "wrap" : "true",
        },
        "opcode" : "sub",
        "plastic" : plastic,
        "learning rate" : learning_rate,
        "max weight" : "0.5",
        "weight config" : {
    #        "type" : "power law",
    #        "exponent" : "1.5",
            "type" : "flat",
            "weight" : base_weight,
            "fraction" : "1.0"
        },
        "myelinated" : "false"
    }

    # Create connections
    connections = [
        exc_exc,
        exc_inh,
        inh_exc,
    ]

    modules = []
    if visualizer:
        modules = [
            {
                "type" : "visualizer",
                "layers" : [
                    { "structure" : "snn", "layer" : "exc" },
                    { "structure" : "snn", "layer" : "inh" },
                ]
            },
            {
                "type" : "heatmap",
                "window" : 1000000, # Long term
                "linear" : "true",
                "layers" : [
                    { "structure" : "snn", "layer" : "exc" },
                    { "structure" : "snn", "layer" : "inh" },
                ]
            },
            {
                "type" : "heatmap",
                "window" : 100, # Short term
                "linear" : "false",
                "layers" : [
                    { "structure" : "snn", "layer" : "exc" },
                    { "structure" : "snn", "layer" : "inh" },
                ]
            },
        ]

    env = Environment({"modules" : modules})

# Create network
    network = Network(
        {"structures" : [structure],
         "connections" : connections})

    if infile is not None:
        if not path.exists(infile):
            print("Could not open state file: " + infile)
        else:
            print("Loading state from " + infile + " ...")
            network.load_state(infile)
            print("... done.")

    pre_exc_exc_matrix = network.get_weight_matrix("exc exc matrix").to_list()
    pre_exc_inh_matrix = network.get_weight_matrix("exc inh matrix").to_list()
    pre_inh_exc_matrix = network.get_weight_matrix("inh exc matrix").to_list()

    gpus = get_gpus()
    if len(gpus) > 1:
        device = gpus[0 if leaky else 1]
        #device = gpus
    elif len(gpus) == 1:
        device = gpus[0]
    else:
        device = get_cpu()

    if not silent:
        print(network.run(env, {"multithreaded" : "true",
                                "worker threads" : "1",
                                "devices" : device,
                                "iterations" : 1000000,
                                "verbose" : "true"}))
        if outfile is not None:
            print("Saving state to " + outfile + " ...")
            network.save_state(outfile)
            print("... done.")

    post_exc_exc_matrix = network.get_weight_matrix("exc exc matrix").to_list()
    post_exc_inh_matrix = network.get_weight_matrix("exc inh matrix").to_list()
    post_inh_exc_matrix = network.get_weight_matrix("inh exc matrix").to_list()

    def print_matrix_stats(matrix):
        non_zero = sum(1 for x in matrix if x >= 0.0001)
        fraction_non_zero = float(non_zero) / len(matrix)
        mat_sum = sum(matrix)
        mat_avg = (mat_sum / non_zero if non_zero > 0 else 0)
        mat_min = (min(x for x in matrix if x >= 0.0001)
                    if non_zero > 0 else min(matrix))
        mat_max = max(matrix)

        print("  Non-zero: %-9d / %-9d (%7.3f%%)"
            % (non_zero, len(matrix), 100.0 * fraction_non_zero))
        print("  Sum:      %f" % mat_sum)
        print("  Average:  %9.7f" % mat_avg)
        print("  Min:      %9.7f" % mat_min)
        print("  Max:      %9.7f" % mat_max)
        print("")
        return (non_zero, mat_sum, mat_avg, mat_min, mat_max)

    def compare_matrices(pre_matrix, post_matrix, to_size):
        print("Pre:")
        (pre_non_zero, pre_sum, pre_avg, pre_min, pre_max) \
            = print_matrix_stats(pre_matrix)

        print("Post:")
        (post_non_zero, post_sum, post_avg, post_min, post_max) \
            = print_matrix_stats(post_matrix)

        plt.subplot(1,2,1)
        np_mat = np.mat(pre_matrix)
        np_mat = np_mat.reshape(to_size, len(pre_matrix) / to_size)
        np_mat = np_mat.sum(axis=1).reshape(to_size, 1)
        (hist, bins) = np.histogram(np_mat,
                 bins=np.logspace(np.log10(np_mat.min()-0.00001), np.log10(np_mat.max()+0.00001), 35))
        x = [10 ** (0.5 * (np.log10(bins[i]) + np.log10(bins[i+1]))) for i in range(len(hist))]
        hist = [float(v) / hist.sum() for v in hist]
        plt.loglog(x, hist, "o", color='b')

        np_mat = np.mat(post_matrix)
        np_mat = np_mat.reshape(to_size, len(post_matrix) / to_size)
        np_mat = np_mat.sum(axis=1).reshape(to_size, 1)
        (hist, bins) = np.histogram(np_mat,
                 bins=np.logspace(np.log10(np_mat.min()-0.00001), np.log10(np_mat.max()+0.00001), 35))
        x = [10 ** (0.5 * (np.log10(bins[i]) + np.log10(bins[i+1]))) for i in range(len(hist))]
        hist = [float(v) / hist.sum() for v in hist]
        plt.loglog(x, hist, "o", color='r')


        plt.subplot(1,2,2)
        np_mat = np.mat([x for x in pre_matrix if x > 0.0])
        np_mat = np_mat.reshape(np_mat.size, 1)
        (hist, bins) = np.histogram(np_mat,
                 bins=np.logspace(np.log10(np_mat.min()-0.00001), np.log10(np_mat.max()+0.00001), 35))
        x = [10 ** (0.5 * (np.log10(bins[i]) + np.log10(bins[i+1]))) for i in range(len(hist))]
        hist = [float(v) / hist.sum() for v in hist]
        plt.loglog(x, hist, "o", color='b')

        np_mat = np.mat([x for x in post_matrix if x > 0.0])
        np_mat = np_mat.reshape(np_mat.size, 1)
        (hist, bins) = np.histogram(np_mat,
                 bins=np.logspace(np.log10(np_mat.min()-0.00001), np.log10(np_mat.max()+0.00001), 35))
        x = [10 ** (0.5 * (np.log10(bins[i]) + np.log10(bins[i+1]))) for i in range(len(hist))]
        hist = [float(v) / hist.sum() for v in hist]
        plt.loglog(x, hist, "o", color='r')
        plt.show()
        plt.show()

        diff_non_zero = post_non_zero - pre_non_zero
        diff_sum = post_sum - pre_sum
        diff_avg = post_avg - pre_avg
        diff_min = post_min - pre_min
        diff_max = post_max - pre_max

        print("Diff:")
        print("  Non-zero:     %-9d" % diff_non_zero)
        print("  Sum:          %f" % diff_sum)
        print("  Average:      %10.7f" % diff_avg)
        print("  Min:          %10.7f" % diff_min)
        print("  Max:          %10.7f" % diff_max)
        print("")
        print("")

    exc_dim = dim**2
    inh_dim = (dim/2)**2
    print("Excitatory-excitatory Matrix:")
    compare_matrices(pre_exc_exc_matrix, post_exc_exc_matrix, exc_dim)
    print("Excitatory-inhibitory Matrix:")
    compare_matrices(pre_exc_inh_matrix, post_exc_inh_matrix, inh_dim)
    print("Inhibitory-excitatory Matrix:")
    compare_matrices(pre_inh_exc_matrix, post_inh_exc_matrix, exc_dim)

    # Delete the objects
    del network
    del env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str,
                        help='source state file')
    parser.add_argument('-o', type=str,
                        help='destination state file')
    parser.add_argument('-s', action='store_true', default=False,
                        dest='silent',
                        help='just print statistics')
    parser.add_argument('-v', action='store_true', default=False,
                        dest='visualizer',
                        help='run the visualizer')
    args = parser.parse_args()

    main(args.i, args.o, args.silent, args.visualizer)
