from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_gpus, get_cpu, get_mpi_size, get_mpi_rank
from syngen import set_suppress_output, set_warnings, set_debug

from random import random
from os import path
import sys
import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

def build_rc(name, device, fraction, res_dim, num_recurrent):
    # Create main structure
    structure = {"name" : name, "type" : "parallel"}

    reservoir = {
        "name" : "reservoir",
        "neural model" : "nvm_tanh",
        "rows" : res_dim,
        "columns" : res_dim,
        "device" : device,
        "init config" : {
            "type" : "uniform",
            "min" : -1.0,
            "max" : 1.0,
        },
    }


    # Add layers to structure
    structure["layers"] = [ reservoir ]

    # Create connections
    connections = []
    for i in range(num_recurrent):
        connections.append({
            "name" : "r_r_%4d" % i,
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
        })

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
                        "from layer" : "reservoir",
                        "to structure" : s2["name"],
                        "to layer" : "reservoir",
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

def build_environment(structures, visualizer=False):
    modules = []
    if visualizer:
        modules = [
            {
                "type" : "visualizer",
                "colored" : False,
                "layers" : [
                    { "structure" : s["name"], "layer" : l["name"] }
                        for s in structures for l in s["layers"]
                ]
            },
        ]

    return modules

def main(infile=None, outfile=None,
        visualizer=False, refresh_rate=0, device=None,
        iterations=1000000, worker_threads=4,
        engine_multithreading=True):

    num_rcs = 16
    res_dim = 64
    fraction = 1.0
    num_recurrent = 1

    if type(device) is list:
        devices = [device[i % len(device)] for i in range(num_rcs)]
    else:
        devices = [device for _ in range(num_rcs)]

    structures = []
    connections = []
    for i in range(num_rcs):
        name = "rc%05d" % i
        s, c = build_rc(name, devices[i], fraction, res_dim, num_recurrent)
        structures.append(s)
        connections += c

    macro_fraction = 1.0
    micro_fraction = 1.0
    connections += link_rc(structures, res_dim, macro_fraction, micro_fraction)

    # Layers that need ghost copies for MPI
    from_ghosts = set()
    to_ghosts = dict()
    mpi_tags = dict()
    mpi_owners = dict()

    mpi_size = get_mpi_size()
    mpi_rank = get_mpi_rank()

    # Prune network based on MPI rank
    if get_mpi_size() > 1:

        # Round robin the structures
        mpi_tag = 0
        for i,s in enumerate(structures):
            owner = i % mpi_size
            s["mpi owner"] = owner

            # Assign unique tags to layers
            for l in s["layers"]:
                mpi_tags[(s["name"], l["name"])] = mpi_tag
                mpi_owners[(s["name"], l["name"])] = owner
                mpi_tag += 1

        # Identify bridging connections and ghost layers
        filtered_connections = []
        for i,c in enumerate(connections):
            from_owner = mpi_owners[(c["from structure"], c["from layer"])]
            to_owner   = mpi_owners[(c["to structure"],   c["to layer"])]

            # Mark connection and add layers to ghosts if necessary
            if to_owner == mpi_rank:
                if from_owner != mpi_rank:
                    from_ghosts.add((c["from structure"], c["from layer"], from_owner))
                    c["from layer"] = "%s_%s" % (c["from structure"], c["from layer"])
                    c["from structure"] = "ghost"
                filtered_connections.append(c)
            elif from_owner == mpi_rank:
                to_ghosts.setdefault(
                    (c["from structure"], c["from layer"]), set()).add(to_owner)

        # Create ghost layers and structure
        ghost_layers = []
        for s in structures:
            for l in s["layers"]:
                for s_name, l_name, from_owner in from_ghosts:
                    if s["name"] == s_name and l["name"] == l_name:
                        gl = deepcopy(l)
                        gl["name"] = "%s_%s" % (s["name"], l["name"])
                        gl["structure"] = "ghost"
                        gl["ghost"] = True
                        ghost_layers.append(gl)

        ghost_structure = {
            "name" : "ghost", "type" : "parallel", "layers" : ghost_layers}

        structures = [s for s in structures if s["mpi owner"] == mpi_rank]
        structures.append(ghost_structure)
        connections = filtered_connections

    network = Network(
        {"structures" : structures,
         "connections" : connections})

    #modules = build_environment(structures, visualizer)
    modules = build_environment(structures,
        visualizer if mpi_rank == 0 else False)

    # Add MPI modules for ghost layer transfers
    if len(from_ghosts) > 0 or len(to_ghosts) > 0:
        modules.append({
            "type" : "mpi",
            "layers" : [
                {
                    "structure" : "ghost",
                    "layer" : "%s_%s" % (s_name, l_name),
                    "input" : True,
                    "mpi source" : from_owner,
                    "mpi tag" : mpi_tags[(s_name, l_name)],
                } for s_name,l_name,from_owner in from_ghosts
            ] + [
                {
                    "structure" : s_name,
                    "layer" : l_name,
                    "output" : True,
                    "lockstep" : True,
                    "mpi destinations" : list(to_owners),
                    "mpi tag" : mpi_tags[(s_name, l_name)],
                } for (s_name,l_name,),to_owners in to_ghosts.iteritems()
            ]
        })
    
        import json
        print(json.dumps(modules[-1], indent=4))

    env = Environment({"modules" : modules})

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
        devices = get_gpus() + [get_cpu()]
    elif args.host or len(get_gpus()) == 0:
        device = get_cpu()
    elif args.gpus:
        device = get_gpus()
    else:
        device = get_gpus()[args.d]

    #device = get_gpus()

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    main(args.i, args.o, args.visualizer, args.r, device, args.it, args.w, args.e)
