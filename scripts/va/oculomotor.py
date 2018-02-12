from syngen import Network, Environment
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug

from os import path
import sys
import argparse

def build_exc_inh_pair(
        exc_name,
        inh_name,
        rows,
        cols,

        half_inh = True,
        mask = True,

        exc_tau = 0.05,
        inh_tau = 0.05,

        exc_decay = 0.02,
        inh_decay = 0.02,

        exc_noise_rate = 1,
        inh_noise_rate = 0,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = 31,
        exc_inh_rf = 123,
        inh_exc_rf = 83,
        inh_inh_rf = 63,

        mask_rf = 31,

        exc_exc_fraction = 1,
        exc_inh_fraction = 1,
        inh_exc_fraction = 1,
        inh_inh_fraction = 1,

        exc_exc_mean = 0.05,
        exc_inh_mean = 0.025,
        inh_exc_mean = 0.025,
        inh_inh_mean = 0.025,

        exc_exc_std_dev = 0.01,
        exc_inh_std_dev = 0.005,
        inh_exc_std_dev = 0.005,
        inh_inh_std_dev = 0.005):

    exc_noise_strength = 0.5 / exc_tau
    inh_noise_strength = 0.5 / inh_tau
    exc = {
        "name" : exc_name,
        "neural model" : "oscillator",
        "rows" : rows,
        "columns" : cols,
        "tau" : exc_tau,
        "decay" : exc_decay,
        "noise config" : {
            "type" : "poisson",
            "value" : exc_noise_strength,
            "rate" : exc_noise_rate,
            "random" : exc_noise_random
        }}
    inh = {
        "name" : inh_name,
        "neural model" : "oscillator",
        "rows" : rows/2 if half_inh else rows,
        "columns" : cols/2 if half_inh else cols,
        "tau" : inh_tau,
        "decay" : inh_decay,
        "noise config" : {
            "type" : "poisson",
            "value" : inh_noise_strength,
            "rate" : inh_noise_rate,
            "random" : inh_noise_random
        }}

    connections = [
        {
            "from layer" : exc_name,
            "to layer" : exc_name,
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
                "wrap" : "false"
            }
        },
        {
            "from layer" : exc_name,
            "to layer" : inh_name,
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
                ] if mask else [ { } ]
            },
            "arborized config" : {
                "field size" : exc_inh_rf,
                "stride" : 2 if half_inh else 1,
                "wrap" : "false"
            }
        },
        {
            "from layer" : inh_name,
            "to layer" : exc_name,
            "type" : "divergent" if half_inh else "convergent",
            "convolutional" : "true",
            "opcode" : "sub",
            "plastic" : "false",
            "weight config" : {
                "type" : "gaussian",
                "mean" : inh_exc_mean,
                "std dev" : inh_exc_std_dev,
                "fraction" : inh_exc_fraction,
                "circular mask" : [ { } ] if not half_inh else None
            },
            "arborized config" : {
                "field size" : inh_exc_rf,
                "stride" : 2 if half_inh else 1,
                "wrap" : "false"
            }
        },
#        {
#            "from layer" : inh_name,
#            "to layer" : inh_name,
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
#                "wrap" : "false"
#            }
#        },
    ]

    return [exc, inh], connections


def build_network(dim=200):
    # Create main structure (parallel engine)
    structure = {"name" : "oculomotor", "type" : "parallel"}

    # Add retinal layer
    vision_layer = {
        "name" : "vision",
        "neural model" : "oscillator",
        "rows" : dim,
        "columns" : dim}

    sc_layers, sc_conns = build_exc_inh_pair(
        "sc_exc", "sc_inh",
        dim, dim,
        half_inh = True,
        mask = True,

        exc_tau = 0.05,
        inh_tau = 0.1,

        exc_decay = 0.02,
        inh_decay = 0.05,

        exc_noise_rate = 1,
        inh_noise_rate = 0,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = 31,
        exc_inh_rf = 123,
        inh_exc_rf = 83,
        inh_inh_rf = 63,

        mask_rf = 31,

        exc_exc_fraction = 1,
        exc_inh_fraction = 1,
        inh_exc_fraction = 1,
        inh_inh_fraction = 1,

        exc_exc_mean = 0.05,
        exc_inh_mean = 0.025,
        inh_exc_mean = 0.025,
        inh_inh_mean = 0.025,

        exc_exc_std_dev = 0.01,
        exc_inh_std_dev = 0.005,
        inh_exc_std_dev = 0.005,
        inh_inh_std_dev = 0.005)

    scale = 20
    sc_out_layers, sc_out_conns = build_exc_inh_pair(
        "sc_out_exc", "sc_out_inh",
        scale, scale,
        half_inh = True,
        mask = True,

        exc_tau = 0.1,
        inh_tau = 0.2,

        exc_decay = 0.05,
        inh_decay = 0.1,

        exc_noise_rate = 1,
        inh_noise_rate = 0,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = 3,
        exc_inh_rf = 11,
        inh_exc_rf = 9,
        inh_inh_rf = 7,

        mask_rf = 3,

        exc_exc_fraction = 1,
        exc_inh_fraction = 1,
        inh_exc_fraction = 1,
        inh_inh_fraction = 1,

        exc_exc_mean = 0.05,
        exc_inh_mean = 0.05,
        inh_exc_mean = 0.05,
        inh_inh_mean = 0.05,

        exc_exc_std_dev = 0.01,
        exc_inh_std_dev = 0.01,
        inh_exc_std_dev = 0.01,
        inh_inh_std_dev = 0.01)


    # Add layers to structure
    structure["layers"] = [vision_layer] + sc_layers + sc_out_layers

    # Create connections
    receptive_field = 31
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
            },
            "arborized config" : {
                "field size" : receptive_field,
                "stride" : 1,
                "wrap" : "false"
            }
        },
        {
            "from layer" : "sc_exc",
            "to layer" : "sc_out_exc",
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.1,
            },
            "arborized config" : {
                "field size" : dim/scale,
                "stride" : dim/scale,
                "wrap" : "false",
                "offset" : 0
            }
        }] + sc_conns + sc_out_conns

    # Create network
    return Network(
        {"structures" : [structure],
         "connections" : connections})

def build_environment(visualizer=False):
    # Create environment modules
    modules = [
        {
            "type" : "gaussian_random_input",
            "rate" : "1000",
            "std dev" : "5",
            "value" : "0.1",
            "normalize" : "true",
            "peaks" : "100",
            "random" : "true",
            "layers" : [
                {
                    "structure" : "oculomotor",
                    "layer" : "vision"
                }
            ]
        }
    ]
    if visualizer:
        modules.append({
            "type" : "visualizer",
            "layers" : [
                { "structure" : "oculomotor", "layer" : "vision" },
                { "structure" : "oculomotor", "layer" : "sc_exc" },
                { "structure" : "oculomotor", "layer" : "sc_inh" },
                { "structure" : "oculomotor", "layer" : "sc_out_exc" },
                { "structure" : "oculomotor", "layer" : "sc_out_inh" }
            ]
        })
        modules.append({
            "type" : "heatmap",
            "window" : "1000",
            "linear" : "true",
            "layers" : [
                { "structure" : "oculomotor", "layer" : "vision" },
                { "structure" : "oculomotor", "layer" : "sc_exc" },
                { "structure" : "oculomotor", "layer" : "sc_inh" },
                { "structure" : "oculomotor", "layer" : "sc_out_exc" },
                { "structure" : "oculomotor", "layer" : "sc_out_inh" }
            ]
        })

    return Environment({"modules" : modules})

def main(infile=None, outfile=None, do_training=True,
        visualizer=False, device=None, rate=0, iterations=1000000):
    dim = 200

    network = build_network(dim)
    env = build_environment(visualizer)

    network.save("networks/ocm.json")
    env.save("environments/ocm.json")

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

    if args.host or len(get_gpus()) == 0:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    main(args.i, args.o, args.train, args.visualizer, device, args.r)
