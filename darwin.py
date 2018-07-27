#!/usr/bin/env python3

import os
import sys
import time
import argparse
import itertools as it
import numpy as np

import cgp
import filehandle as fh

argparser = argparse.ArgumentParser()

argparser.add_argument("circuit", help = "file containing verilog of circuit to be optimized")
argparser.add_argument("-seed", type = int, default = 0, help = "random seed (-1 means complete random)")

argparser.add_argument("-mutation", type = float, default = 1.0, help = "gene mutation rate by node")
argparser.add_argument("-lambda", type = int, default = 1, help = "lambda value (children count)", dest = "popsize")
argparser.add_argument("-kappa", type = float, default = 1.0, help = "number of cols in CGP matrix by nodes on seed circuit")

argparser.add_argument("-no-neutral-drift", action = "store_false", dest = "neutral_drift", help = "disables neutral drift")

argparser.add_argument("-functions", type = str.upper, default = ( "AND", "OR", "MAJ", "NOT" ),
                       nargs = "*", help = "available functions", choices = ( "AND", "OR", "MAJ", "NOT" ))

argparser.add_argument("-fitness", type = float, default = -np.inf, help = "min fitness")
argparser.add_argument("-generations", type = int, default = np.inf, help = "max generations")

argparser.add_argument("-save-curve", help = "save curve for best individuals into file (npz format)")
argparser.add_argument("-save-stats", help = "save generations stats into file (npz format)")
argparser.add_argument("-save-best", help = "save best individual into file (npz format)")
argparser.add_argument("-save-time", help = "save runtime into file (npz format)")
argparser.add_argument("-save-energy", help = "save final energy into file (npz format)")
argparser.add_argument("-save-size", help = "save final size into file (npz format)")
argparser.add_argument("-save-depth", help = "save final depth into file (npz format)")
argparser.add_argument("-save-times", help = "save runtime per generation file (npz format)")

argparser.add_argument("-plot", help = "plot best individuals curve into file")
argparser.add_argument("-verilog", help = "convert the best individual into verilog")
argparser.add_argument("-module-name", help = "verilog's module name (defaults to input file's module name)")

argparser.add_argument("-log", type = fh.Log, default = fh.log, help = "file where to print logging")
argparser.add_argument("-quiet", action = "store_const", const = fh.null, dest = "log", help = "disable logging")
argparser.add_argument("-merge", type = int, default = 0, help = "merges multiple generations into one line (0 merges all)")
argparser.add_argument("-print-on-change", action = "store_true", help = "merges when fitness changes.")

argparser.add_argument("-raise-keyboard", action = "store_true", help = "raises keyboard interrupt at end if interrupted during evolution")
argparser.add_argument("-no-save-incomplete", action = "store_false", dest = "save_incomplete", help = "do not save data when keyboard interrupt was raised")

def main (argv):
    np.seterr(all = "raise")

    args = argparser.parse_args(argv)
    seed = None if args.seed == -1 else args.seed

    gp = cgp.CGP(args.popsize, args.functions, args.neutral_drift, seed)

    raised = None
    # Enforce kappa to be >= 1
    kappa = max(1.0, args.kappa)

    start_time = time.time()
    # Initialize the CGP
    gp.setup(args.circuit, kappa, args.mutation, args.log)

    try:
        # Run the CGP
        gp.run_until((lambda gp: (
            gp.best_fit.item(0) < args.fitness or
            gp.generation >= args.generations
        )), args.merge, args.print_on_change, log = args.log)

    except KeyboardInterrupt as e:
        # Treat interruptions
        print(file = args.log)

        if not args.save_incomplete:
            # Raise only if required to
            raise

        raised = e

    run_time = time.time() - start_time

    energy = gp.best_fit
    size = np.count_nonzero(gp.best_act)

    print("energy = {:.5f}, size = {}, time = {:.8f}s".format(
        energy, size, run_time
    ), file = args.log)

    # Save data
    if args.save_energy is not None:
        with open(args.save_energy, "wb") as file:
            np.savez_compressed(file, energy = energy)

    if args.save_size is not None:
        with open(args.save_size, "wb") as file:
            np.savez_compressed(file, size = size)

    if args.save_time is not None:
        with open(args.save_time, "wb") as file:
            np.savez_compressed(file, time = run_time)

    if args.save_curve is not None:
        gp.save_curve(args.save_curve)

    if args.save_stats is not None:
        gp.save_stats(args.save_stats)

    if args.save_best is not None:
        gp.save_best(args.save_best)

    if args.plot is not None:
        gp.plot(args.plot)

    if args.verilog is not None:
        with open(args.verilog, "w") as file:
            for line in gp.verilog(gp.best_gen, gp.best_act, args.module_name):
                print(line, file = file)

    if raised is not None:
        if args.raise_keyboard or not isinstance(raised, KeyboardInterrupt):
            # Raise if required to or not KeyboardInterrupt
            raise raised

    return gp

if __name__ == "__main__":
    main(sys.argv[ 1 : ])
