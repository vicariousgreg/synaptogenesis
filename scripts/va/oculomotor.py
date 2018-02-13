from syngen import Network, Environment
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug

from os import path
import sys
import argparse

def build_exc_inh_pair(
        exc_name,
        inh_name,
        rows = 200,
        cols = 200,

        half_inh = True,
        mask = True,

        exc_tau = 0.05,
        inh_tau = 0.05,

        exc_decay = 0.02,
        inh_decay = 0.02,

        exc_noise_rate = 1,
        inh_noise_rate = 1,
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


def build_network(rows=200, cols=200, scale=5):
    dim = min(rows, cols)

    # Create main structure (parallel engine)
    structure = {"name" : "oculomotor", "type" : "parallel"}

    # Add retinal layer
    vision_layer = {
        "name" : "vision",
        "neural model" : "oscillator",
        "rows" : rows,
        "columns" : cols}

    sc_layers, sc_conns = build_exc_inh_pair(
        "sc_exc", "sc_inh",
        rows, cols,
        half_inh = True,
        mask = True,

        exc_tau = 0.05,
        inh_tau = 0.1,

        exc_decay = 0.02,
        inh_decay = 0.05,

        exc_noise_rate = 1,
        inh_noise_rate = 1,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = dim/7,
        exc_inh_rf = dim/2,
        inh_exc_rf = dim/2.5,
        inh_inh_rf = dim/3.5,

        mask_rf = dim/7,

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

    motor_rows = int(rows/scale)
    motor_cols = int(cols/scale)
    motor_dim = min(motor_rows, motor_cols)
    sc_out_layers, sc_out_conns = build_exc_inh_pair(
        "sc_out_exc", "sc_out_inh",
        motor_rows, motor_cols,
        half_inh = True,
        mask = True,

        exc_tau = 0.01,
        inh_tau = 0.1,

        exc_decay = 0.02,
        inh_decay = 0.02,

        exc_noise_rate = 0,
        inh_noise_rate = 0,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = 3,
        exc_inh_rf = motor_dim/2,
        inh_exc_rf = motor_dim/2.5,
        inh_inh_rf = motor_dim/3.5,

        mask_rf = 3,

        exc_exc_fraction = 1,
        exc_inh_fraction = 1,
        inh_exc_fraction = 1,
        inh_inh_fraction = 1,

        exc_exc_mean = 0.1,
        exc_inh_mean = 0.05,
        inh_exc_mean = 0.05,
        inh_inh_mean = 0.05,

        exc_exc_std_dev = 0.02,
        exc_inh_std_dev = 0.01,
        inh_exc_std_dev = 0.01,
        inh_inh_std_dev = 0.01)

    gating_layer = {
        "name" : "gating",
        "neural model" : "oscillator",
        "rows" : motor_rows,
        "columns" : motor_cols,
        "tau" : 0.5,
        "decay" : 0.5,
        "tonic" : 0.0}

    # Add layers to structure
    structure["layers"] = [vision_layer] + \
        sc_layers + sc_out_layers + [gating_layer]

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
                "field size" : rows/motor_rows,
                "stride" : cols/motor_cols,
                "wrap" : "false",
                "offset" : 0
            }
        },
        {
            "from layer" : "gating",
            "to layer" : "sc_out_exc",
            "type" : "one to one",
            "opcode" : "mult",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 1.0,
            }
        }
        ] + sc_conns + sc_out_conns

    # Create network
    return Network(
        {"structures" : [structure],
         "connections" : connections})

def build_environment(rows=200, cols=200, scale=5, visualizer=False):
    dim = min(rows, cols)
    motor_dim = min(rows/scale, cols/scale)

    # Create environment modules
    modules = [
        {
            "type" : "gaussian_random_input",
            "rate" : "1000",
            "border" : dim/10,
            "std dev" : dim/40,
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
        },
        {
            "type" : "gaussian_random_input",
            "rate" : "1000",
            "border" : motor_dim/10,
            "std dev" : motor_dim/5,
            "value" : "0.5",
            "normalize" : "true",
            "peaks" : "1",
            "random" : "false",
            "layers" : [
                {
                    "structure" : "oculomotor",
                    "layer" : "gating"
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
                { "structure" : "oculomotor", "layer" : "sc_out_inh" },
                { "structure" : "oculomotor", "layer" : "gating" },
            ]
        })
        modules.append({
            "type" : "heatmap",
            "stats" : "false",
            "window" : "1000",
            "linear" : "true",
            "layers" : [
                { "structure" : "oculomotor", "layer" : "vision" },
                { "structure" : "oculomotor", "layer" : "sc_exc" },
                { "structure" : "oculomotor", "layer" : "sc_inh" },
                { "structure" : "oculomotor", "layer" : "sc_out_exc" },
                { "structure" : "oculomotor", "layer" : "sc_out_inh" },
                { "structure" : "oculomotor", "layer" : "gating" },
            ]
        })

    return Environment({"modules" : modules})

def main(infile=None, outfile=None, do_training=True,
        visualizer=False, device=None, rate=0, iterations=1000000):
    rows = 100
    cols = 200
    scale = 5

    network = build_network(rows, cols, scale)
    env = build_environment(rows, cols, scale, visualizer)

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
