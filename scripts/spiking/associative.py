from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug

from os import path
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

def build_module(structure, dim=64):
    inh_fraction = 1

    # Neuron parameters
    exc_init = "regular"
    inh_init = "fast"
    #exc_init = "random positive"
    #inh_init = "random negative"

    # Plasticity parameters
    plastic = True
    stp = True
    stp_tau = "5000"
    exc_learning_rate = 0.001
    inh_learning_rate = 0.001

    # Noise Parameters
    exc_noise_strength = 10.0
    exc_noise_rate = 10
    inh_noise_strength = 10.0
    inh_noise_rate = 0
    exc_random = False
    inh_random = False

    # Weight parameters
    init = "flat"
    myelinated = False

    exc_exc_weight_init = init
    exc_exc_exponent = 1.5
    exc_exc_base_weight = 0.005
    exc_exc_base_weight_min = 0.00011
    exc_exc_base_weight_max = 0.2
    exc_exc_fraction = 0.1

    exc_inh_weight_init = init
    exc_inh_exponent = 1.5
    exc_inh_base_weight = 0.005
    exc_inh_base_weight_min = 0.00011
    exc_inh_base_weight_max = 0.2
    exc_inh_fraction = 0.1

    inh_exc_weight_init = init
    inh_exc_exponent = 1.5
    inh_exc_base_weight = 0.01
    inh_exc_base_weight_min = 0.00011
    inh_exc_base_weight_max = 0.2
    inh_exc_fraction = 1.0

    exc_exc_spread = 25
    exc_inh_spread = 9
    inh_exc_spread = 15
    exc_inh_mask = 3

    # Excitatory layer
    excitatory = {
        "name" : "exc",
        "neural model" : "izhikevich",
        "rows" : dim,
        "columns" : dim,
        "neuron spacing" : 0.1,
        "init" : exc_init
    }

    # Inhibitory layer
    inhibitory = {
        "name" : "inh",
        "neural model" : "izhikevich",
        "rows" : dim/inh_fraction,
        "columns" : dim/inh_fraction,
        "neuron spacing" : 0.1 * inh_fraction,
        "init" : inh_init
    }

    # Noise
    if exc_noise_rate > 0 and exc_noise_strength > 0:
        excitatory["init config"] = {
            "type" : "poisson",
            "value" : exc_noise_strength,
            "rate" : exc_noise_rate,
            "random" : exc_random
        }
    if inh_noise_rate > 0 and inh_noise_strength > 0:
        inhibitory["init config"] = {
            "type" : "poisson",
            "value" : inh_noise_strength,
            "rate" : inh_noise_rate,
            "random" : inh_random
        }

    exc_exc = {
        "from structure" : structure,
        "from layer" : "exc",
        "to structure" : structure,
        "to layer" : "exc",
        "name" : "exc exc matrix",
        "type" : "convergent",
        "arborized config" : {
            "field size" : exc_exc_spread,
            "wrap" : True,
        },
        "opcode" : "add",
        "plastic" : plastic,
        "sparse" : True,
        "learning rate" : exc_learning_rate,
        "max weight" : "0.5",
        "myelinated" : myelinated,
        "short term plasticity" : stp,
        "short term plasticity tau" : stp_tau
    }

    exc_inh = {
        "from structure" : structure,
        "from layer" : "exc",
        "to structure" : structure,
        "to layer" : "inh",
        "name" : "exc inh matrix",
        "type" : "convergent",
        "arborized config" : {
            "field size" : exc_inh_spread,
            "stride" : inh_fraction,
            "wrap" : True,
        },
        "opcode" : "add",
        "plastic" : plastic,
        "sparse" : True,
        "learning rate" : exc_learning_rate,
        "max weight" : "0.5",
        "myelinated" : myelinated,
        "short term plasticity" : stp,
        "short term plasticity tau" : stp_tau
    }
    inh_exc = {
        "from structure" : structure,
        "from layer" : "inh",
        "to structure" : structure,
        "to layer" : "exc",
        "name" : "inh exc matrix",
        "type" : "divergent",
        "arborized config" : {
            "field size" : inh_exc_spread,
            "stride" : inh_fraction,
            "wrap" : True,
        },
        "opcode" : "sub",
        "plastic" : plastic,
        "sparse" : True,
        "learning rate" : inh_learning_rate,
        "max weight" : "0.5",
        "myelinated" : myelinated,
        "short term plasticity" : stp,
        "short term plasticity tau" : stp_tau
    }

    # Exc Exc init
    if exc_exc_weight_init == "zero":
        exc_exc["weight config"] = {
                "type" : "flat",
                "weight" : 0.00011,
                "fraction" : exc_exc_fraction,
                "diagonal" : False,
                "circular mask" : {}
            }
    elif exc_exc_weight_init == "random":
        exc_exc["weight config"] = {
                "type" : "random",
                "min weight" : exc_exc_base_weight_min,
                "max weight" : exc_exc_base_weight_max,
                "fraction" : exc_exc_fraction,
                "diagonal" : False,
                "circular mask" : {}
            }
    elif exc_exc_weight_init == "flat":
        exc_exc["weight config"] = {
                "type" : "flat",
                "weight" : exc_exc_base_weight,
                "fraction" : exc_exc_fraction,
                "diagonal" : False,
                "circular mask" : {}
            }
    elif exc_exc_weight_init == "power law":
        exc_exc["weight config"] = {
                "type" : "power law",
                "exponent" : exc_exc_exponent,
                "min weight" : exc_exc_base_weight_min,
                "max weight" : exc_exc_base_weight_max,
                "fraction" : exc_exc_fraction,
                "diagonal" : False,
                "circular mask" : {}
            }

    # Exc Inh init
    if exc_inh_weight_init == "zero":
        exc_inh["weight config"] = {
                "type" : "flat",
                "weight" : 0.00011,
                "fraction" : exc_inh_fraction,
                "circular mask" : [
                    {
                        "diameter" : exc_inh_mask,
                        "invert" : True
                    },
                    { }
                ]
            }
    elif exc_inh_weight_init == "random":
        exc_inh["weight config"] = {
                "type" : "random",
                "min weight" : exc_inh_base_weight_min,
                "max weight" : exc_inh_base_weight_max,
                "fraction" : exc_inh_fraction,
                "circular mask" : [
                    {
                        "diameter" : exc_inh_mask,
                        "invert" : True
                    },
                    { }
                ]
            }
    elif exc_inh_weight_init == "flat":
        exc_inh["weight config"] = {
                "type" : "flat",
                "weight" : exc_inh_base_weight,
                "fraction" : exc_inh_fraction,
                "circular mask" : [
                    {
                        "diameter" : exc_inh_mask,
                        "invert" : True
                    },
                    { }
                ]
            }
    elif exc_inh_weight_init == "power law":
        exc_inh["weight config"] = {
                "type" : "power law",
                "exponent" : exc_inh_exponent,
                "min weight" : exc_inh_base_weight_min,
                "max weight" : exc_inh_base_weight_max,
                "fraction" : exc_inh_fraction,
                "circular mask" : [
                    {
                        "diameter" : exc_inh_mask,
                        "invert" : True
                    },
                    { }
                ]
            }

    # Inh Exc init
    if inh_exc_weight_init == "zero":
        inh_exc["weight config"] = {
                "type" : "flat",
                "weight" : 0.00011,
                "fraction" : inh_exc_fraction,
                "distance callback" : "gaussian",
            }
    elif inh_exc_weight_init == "random":
        inh_exc["weight config"] = {
                "type" : "random",
                "min weight" : inh_exc_base_weight_min,
                "max weight" : inh_exc_base_weight_max,
                "fraction" : inh_exc_fraction,
                "distance callback" : "gaussian",
            }
    elif inh_exc_weight_init == "flat":
        inh_exc["weight config"] = {
                "type" : "flat",
                "weight" : inh_exc_base_weight,
                "fraction" : inh_exc_fraction,
                "distance callback" : "gaussian",
            }
    elif inh_exc_weight_init == "power law":
        inh_exc["weight config"] = {
                "type" : "power law",
                "exponent" : inh_exc_exponent,
                "min weight" : inh_exc_base_weight_min,
                "max weight" : inh_exc_base_weight_max,
                "fraction" : inh_exc_fraction,
                "distance callback" : "gaussian",
            }

    # Create connections
    connections = [
        exc_exc,
        exc_inh,
        inh_exc,
    ]

    return [excitatory, inhibitory], connections

def build_network(dim=64):
    a_layers, a_connections = build_module("snn_a", dim)
    b_layers, b_connections = build_module("snn_b", dim)

    # Create main structure
    a_structure = {"name" : "snn_a",
                   "type" : "parallel",
                   "layers" : a_layers}
    b_structure = {"name" : "snn_b",
                   "type" : "parallel",
                   "layers" : b_layers}

    bridge_learning_rate = 0.001
    bridge_weight = 0.01
    bridge_fraction = 0.01

    bridge_a_b = {
        "from structure" : "snn_a",
        "from layer" : "exc",
        "to structure" : "snn_b",
        "to layer" : "exc",
        "name" : "bridge a b matrix",
        "type" : "fully connected",
        "opcode" : "add",
        "delay" : 10,
        "plastic" : True,
        "sparse" : True,
        "learning rate" : bridge_learning_rate,
        "max weight" : "0.5",
        "myelinated" : True,
        "short term plasticity" : True,
        "short term plasticity tau" : 5000,
        "weight config" : {
            "type" : "flat",
            "weight" : bridge_weight,
            "fraction" : bridge_fraction
        }
    }

    bridge_b_a = {
        "from structure" : "snn_b",
        "from layer" : "exc",
        "to structure" : "snn_a",
        "to layer" : "exc",
        "name" : "bridge b a matrix",
        "type" : "fully connected",
        "opcode" : "add",
        "delay" : 10,
        "plastic" : True,
        "sparse" : True,
        "learning rate" : bridge_learning_rate,
        "max weight" : "0.5",
        "myelinated" : True,
        "short term plasticity" : True,
        "short term plasticity tau" : 5000,
        "weight config" : {
            "type" : "flat",
            "weight" : bridge_weight,
            "fraction" : bridge_fraction
        }
    }

    return Network(
        {"structures" : [a_structure, b_structure],
         "connections" : a_connections + b_connections + [bridge_a_b, bridge_b_a]})


def build_environment(visualizer=False, peaks=False, std_dev=10):
    modules = []
    if visualizer:
        modules = [
            {
                "type" : "visualizer",
                "colored" : False,
                "layers" : [
                    { "structure" : "snn_a", "layer" : "exc" },
                    { "structure" : "snn_a", "layer" : "inh" },
                    { "structure" : "snn_b", "layer" : "exc" },
                    { "structure" : "snn_b", "layer" : "inh" },
                ]
            },
            {
                "type" : "visualizer",
                "colored" : True,

                "decay" : False,
                "window" : 256,

                #"decay" : True,
                #"window" : 1024,
                #"bump" : 128,

                "layers" : [
                    { "structure" : "snn_a", "layer" : "exc" },
                    { "structure" : "snn_a", "layer" : "inh" },
                    { "structure" : "snn_b", "layer" : "exc" },
                    { "structure" : "snn_b", "layer" : "inh" },
                ]
            },
            {
                "type" : "heatmap",
                "window" : 1000000, # Long term
                "linear" : True,
                "layers" : [
                    { "structure" : "snn_a", "layer" : "exc" },
                    { "structure" : "snn_a", "layer" : "inh" },
                    { "structure" : "snn_b", "layer" : "exc" },
                    { "structure" : "snn_b", "layer" : "inh" },
                ]
            },
            {
                "type" : "heatmap",
                "window" : 1000, # Short term
                "linear" : False,
                "layers" : [
                    { "structure" : "snn_a", "layer" : "exc" },
                    { "structure" : "snn_a", "layer" : "inh" },
                    { "structure" : "snn_b", "layer" : "exc" },
                    { "structure" : "snn_b", "layer" : "inh" },
                ]
            }
        ]

    return Environment({"modules" : modules})

def print_matrix_stats(matrix):
    non_zero = sum(1 for x in matrix if x >= 0.00001)
    fraction_non_zero = float(non_zero) / len(matrix)
    mat_sum = sum(matrix)
    mat_avg = (mat_sum / non_zero if non_zero > 0 else 0)
    mat_min = (min(x for x in matrix if x >= 0.00001)
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

def compare_matrices(init_matrix, pre_matrix, post_matrix, to_size):
    print("Init:")
    (init_non_zero, init_sum, init_avg, init_min, init_max) \
        = print_matrix_stats(init_matrix)

    print("Pre:")
    (pre_non_zero, pre_sum, pre_avg, pre_min, pre_max) \
        = print_matrix_stats(pre_matrix)

    print("Post:")
    (post_non_zero, post_sum, post_avg, post_min, post_max) \
        = print_matrix_stats(post_matrix)


    num_diff = np.sum(np.mat(pre_matrix) != np.mat(post_matrix))
    diff_non_zero = post_non_zero - pre_non_zero
    diff_sum = post_sum - pre_sum
    diff_avg = post_avg - pre_avg
    diff_min = post_min - pre_min
    diff_max = post_max - pre_max

    print("Diff:")
    print("  Num diff:     %-9d" % num_diff)
    print("  Non-zero:     %-9d" % diff_non_zero)
    print("  Sum:          %f" % diff_sum)
    print("  Average:      %10.7f" % diff_avg)
    print("  Min:          %10.7f" % diff_min)
    print("  Max:          %10.7f" % diff_max)
    print("")
    print("")

    def plot_matrix(matrix, color):
        np_mat = np.mat(matrix)
        np_mat = np_mat.reshape(to_size, len(matrix) / to_size)
        np_mat = np_mat.sum(axis=1).reshape(to_size, 1)
        (hist, bins) = np.histogram(np_mat,
                 bins=np.logspace(np.log10(max(0.00001, np_mat.min()-0.00001)),
                             np.log10(np_mat.max()+0.00001), 100))
        x = [10 ** (0.5 * (np.log10(bins[i]) + np.log10(bins[i+1]))) for i in range(len(hist))]
        try:
            hist = [float(v) / hist.sum() for v in hist]
        except ZeroDivisionError:
            pass
        plt.loglog(x, hist, "o", color=color)

    def plot_array(matrix, color):
        np_mat = np.mat([x for x in matrix if x > 0.0])
        if np_mat.size == 0:
            x = 0.0
            hist = [len(matrix)]
        else:
            np_mat = np_mat.reshape(np_mat.size, 1)
            (hist, bins) = np.histogram(np_mat,
                     bins=np.logspace(np.log10(max(0.00001, np_mat.min()-0.00001)),
                         np.log10(np_mat.max()+0.00001), 100))
            x = [10 ** (0.5 * (np.log10(bins[i]) + np.log10(bins[i+1]))) for i in range(len(hist))]
            try:
                hist = [float(v) / hist.sum() for v in hist]
            except ZeroDivisionError:
                pass
        plt.loglog(x, hist, "o", color=color)

    #plt.subplot(1,2,1)
    #plot_matrix(init_matrix, 'g')
    #plot_matrix(pre_matrix, 'b')
    #plot_matrix(post_matrix, 'r')

    #plt.subplot(1,2,2)
    plot_array(init_matrix, 'g')
    plot_array(pre_matrix, 'b')
    plot_array(post_matrix, 'r')

    plt.show()

def main(infile=None, outfile=None, do_training=True, print_stats=True,
        dim=128, peaks=1, visualizer=False, refresh_rate=0, device=None,
        iterations=1000000):
    std_dev = int(dim / 10)

    network = build_network(dim)
    env = build_environment(visualizer, peaks, std_dev)

    if print_stats:
        init_exc_exc_matrix = network.get_weight_matrix("exc exc matrix").to_list()
        init_exc_inh_matrix = network.get_weight_matrix("exc inh matrix").to_list()
        init_inh_exc_matrix = network.get_weight_matrix("inh exc matrix").to_list()
        init_bridge_a_b_matrix = network.get_weight_matrix("bridge a b matrix").to_list()
        init_bridge_b_a_matrix = network.get_weight_matrix("bridge b a matrix").to_list()

    if infile is not None:
        if not path.exists(infile):
            print("Could not open state file: " + infile)
        else:
            print("Loading state from " + infile + " ...")
            network.load_state(infile)
            print("... done.")

    if print_stats:
        pre_exc_exc_matrix = network.get_weight_matrix("exc exc matrix").to_list()
        pre_exc_inh_matrix = network.get_weight_matrix("exc inh matrix").to_list()
        pre_inh_exc_matrix = network.get_weight_matrix("inh exc matrix").to_list()
        pre_bridge_a_b_matrix = network.get_weight_matrix("bridge a b matrix").to_list()
        pre_bridge_b_a_matrix = network.get_weight_matrix("bridge b a matrix").to_list()

    if device is None:
        device = get_gpus()[1]
    if do_training:
        train_args = {"multithreaded" : True,
                      "worker threads" : 4 if device == get_cpu() else 0,
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

    if print_stats:
        post_exc_exc_matrix = network.get_weight_matrix("exc exc matrix").to_list()
        post_exc_inh_matrix = network.get_weight_matrix("exc inh matrix").to_list()
        post_inh_exc_matrix = network.get_weight_matrix("inh exc matrix").to_list()
        post_bridge_a_b_matrix = network.get_weight_matrix("bridge a b matrix").to_list()
        post_bridge_b_a_matrix = network.get_weight_matrix("bridge b a matrix").to_list()

        exc_dim = dim**2
        inh_dim = (dim/2)**2
        bridge_dim = exc_dim ** 2
        print("Excitatory-excitatory Matrix:")
        compare_matrices(init_exc_exc_matrix, pre_exc_exc_matrix, post_exc_exc_matrix, exc_dim)
        print("Excitatory-inhibitory Matrix:")
        compare_matrices(init_exc_inh_matrix, pre_exc_inh_matrix, post_exc_inh_matrix, inh_dim)
        print("Inhibitory-excitatory Matrix:")
        compare_matrices(init_inh_exc_matrix, pre_inh_exc_matrix, post_inh_exc_matrix, exc_dim)
        print("Bridge A-B Matrix:")
        compare_matrices(init_bridge_a_b_matrix, pre_bridge_a_b_matrix, post_bridge_a_b_matrix, bridge_dim)
        print("Bridge B-A Matrix:")
        compare_matrices(init_bridge_b_a_matrix, pre_bridge_b_a_matrix, post_bridge_b_a_matrix, bridge_dim)

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
    parser.add_argument('-dim', type=int, default=128,
                        help='dimensions')
    parser.add_argument('-p', type=int, default=1,
                        help='noise peaks')
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
    args = parser.parse_args()

    if args.host or len(get_gpus()) == 0:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]
        #device = get_gpus()

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    main(args.i, args.o, args.train, args.stats, args.dim, args.p, args.visualizer, args.r, device, args.it)
