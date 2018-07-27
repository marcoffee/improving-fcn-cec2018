import os
import sys
import copy
import time
import textwrap as tw
import itertools as it

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import parser
import filehandle as fh

def compress_size (size):
    """ Compress binary array into integers comprising bits """

    sim_t = np.bool8
    sim_bits = 1

    # Computes the best type to compress boolean data with `size`
    for i in it.count():
        nbytes = 1 << i
        nbits = nbytes << 3

        # If size cannot fit into bits
        if size & (nbits - 1) != 0:
            break

        try:
            # Create numpy dtype
            sim_t = np.dtype("u{}".format(nbytes))
            # Save new bits per type element, e.g., 4 for an integer
            sim_bits = nbits
        except TypeError:
            break

    return sim_t, sim_bits

def binary_combinations (size):
    """ Build all binary combinations of `size` bits """

    exp_size = 1 << size
    dtype, dbits = compress_size(exp_size)
    numbers = (exp_size + dbits - 1) // dbits

    result = np.empty(( size, numbers ), dtype)
    vals = np.arange(exp_size)
    temp = np.empty_like(vals)

    for i in range(size - 1, -1, -1):
        np.bitwise_and(1 << i, vals, temp)
        compress(temp, dtype, dbits, result[i])

    return result

def compress (data, dtype, dbits, out):
    # Convert only if type is not boolean
    if dtype is not np.bool8:
        numbers = (data.shape[0] + dbits - 1) // dbits
        data = data.reshape(numbers, dbits)

        bits_shift = np.arange(dbits, dtype = dtype)
        np.left_shift(data > 0, bits_shift, out = data)

        out = np.bitwise_or.reduce(data, axis = 1, out = out)
    else:
        out[:] = data

dgen = np.int64

class CGPException (Exception):
    """ Class to represent a CGP Exception """
    pass

class CGP (object):
    """ Class to represent the Cartesian Genetic Programming """

    __slots__ = [
        "__p_find_active_loop", "__p_simulate_loop", "__p_energy_loop",
        "__p_arr_nodes", "__p_arr_cols", "__p_arr_size",
        "__p_funcs", "__p_arities", "__p_ffmts", "__p_famap", "__p_feval",
        "__p_integer_mul", "__p_integer_add", "__p___sims_data",
        "__seed", "__random", "__popsize", "__fnames", "__arity",
        "__ni", "__no", "__nc", "__gen", "__act", "__fit", "__generation",
        "__inp_data", "__exp_data", "__best_curve", "__best_gener",
        "__better", "__worse", "__equal", "__same", "__times", "__time",
        "__module", "neutral_drift", "mutate_gene"
    ]

    def init_properties (self):
        """ Initialize CGP's properties """
        self.__p_find_active_loop = self.__p_simulate_loop = None
        self.__p_energy_loop = self.__p_arr_nodes = self.__p_arr_cols = None
        self.__p_arr_size = self.__p_funcs = self.__p_arities = None
        self.__p_ffmts = self.__p_famap = self.__p_feval = None
        self.__p_integer_mul = self.__p_integer_add = None
        self.__p___sims_data = None

    def __init__ (self, popsize, gates = (), neutral_drift = True, seed = 0):
        """ Constructs CGP """
        self.init_properties()

        self.__seed = seed
        self.__popsize = popsize

        self.__fnames = np.sort(gates)
        self.__fnames.setflags(write = False)

        self.__arity = self.arities.max()
        self.__module = "circuit"

        self.neutral_drift = neutral_drift

    def random_genotype (self, amount = 1):
        """ Generate a random genotype """
        return self.to_integer(self.__random.random_sample(
            ( amount, self.nodes_size + self.no )
        ))

    def setup_one_lambda (self, c_seed, kappa, mutation):
        """ Setup 1+lambda ES """

        self.__module, cinps, chromo, couts = parser.convert(c_seed)
        self.__ni = len(cinps)
        self.__no = len(couts)
        self.__nc = int(np.ceil(len(chromo) * kappa))
        self.mutate_gene = mutation / (self.nc + self.no)

        # Generate one base random genotype
        self.__gen = self.random_genotype(1)

        id_map = np.arange(self.ni + len(chromo))

        funs, args, outs = self.break_genotypes(self.__gen)
        gates = { name : i for i, name in enumerate(self.fnames) }

        # Randomly distributes the original nodes into the random genotype
        pieces = np.zeros(self.nc, np.bool8)
        pieces[ : len(chromo) ] = True
        self.__random.shuffle(pieces)

        cpos = 0

        for pos in np.flatnonzero(pieces):
            cfun, *cargs = chromo[cpos]

            if cfun not in gates:
                raise CGPException("Seed circuit has unselected "
                                   "function `{}`.".format(cfun))

            funs[ ... , pos ] = gates[cfun]
            args[ ... , pos , : len(cargs)] = id_map[cargs]
            id_map[cpos + self.ni] = pos + self.ni

            cpos += 1

        outs[:] = id_map[couts]
        self.__act = self.find_active(self.__gen)

        # Generate combinations
        self.__inp_data = binary_combinations(self.ni)
        self.__inp_data.setflags(write = False)

        self.simulate(self.__gen, self.__act, self.__sims_data[ : 1 ])
        self.__exp_data = self.__sims_data[ 0 , -self.no : ].copy()
        self.__exp_data.setflags(write = False)

        self.__fit = self.get_fitness(self.__gen, self.__act, self.__sims_data[ : 1 ])

    def setup (self, c_seed, kappa = 1.0, mutation = 1.0, log = fh.log):
        """ Setup CGP """

        start_time = time.time()

        self.__random = np.random.RandomState(self.__seed)
        self.__generation = 0

        self.setup_one_lambda(c_seed, kappa, mutation)

        self.__best_gener = []
        self.__best_curve = []

        self.__better     = []
        self.__equal      = []
        self.__worse      = []

        self.__same       = []
        self.__times      = []
        self.__time       = 0.0

        rtime = time.time() - start_time

        self.store_data(0, 0, 0, 0, rtime)
        self.print_debug(log)

    def store_data (self, bet, equ, wor, same, rtime):
        """ Stores newly generated data into the class """

        best_fit = self.best_fit

        if not self.__best_curve or best_fit != self.__best_curve[-1]:
            # Add to curve iff changed
            self.__best_curve.append(best_fit)
            self.__best_gener.append(self.generation)

        self.__better.append(bet)
        self.__equal.append(equ)
        self.__worse.append(wor)

        self.__same.append(same)
        self.__times.append(rtime)
        self.__time += rtime

    def one_lambda (self):
        """ Run 1+lambda ES """

        # Get current best
        best_gen, best_act, best_fit = self.best

        # Storage for new individuals
        off_gen = np.empty(( self.popsize, self.genotype_size ), best_gen.dtype)
        off_act = np.empty(self.popsize, np.object)
        off_fit = np.empty(self.popsize, np.object)

        same = 0
        sames = np.zeros(self.popsize, np.bool8)
        diff = []

        # Generate individuals
        for pos in range(self.popsize):
            gen = self.mutation(best_gen)
            off_gen[pos] = gen

            if self.same_critical(best_gen, best_act, gen):
                off_act[pos] = best_act
                off_fit[pos] = best_fit
                sames[pos] = True
                same += 1
            else:
                diff.append(pos)

        # Evaluate different individuals
        if diff:
            diff = np.array(diff)

            gen = off_gen[diff]
            act = self.find_active(gen)
            sim = self.simulate(gen, act, self.__sims_data[ : len(diff) ])
            fit = self.get_fitness(gen, act, sim)

            for i, pos in enumerate(diff):
                off_act[pos] = act[i]
                off_fit[pos] = fit[i]

        pos = None
        better = equal = worse = 0
        new_fit = best_fit

        # Compare individuals to parent
        for i, fit in enumerate(off_fit):
            if fit < best_fit:
                better += 1
            elif best_fit < fit:
                worse += 1
            else:
                equal += 1

            change = fit <= new_fit if self.neutral_drift else fit < new_fit

            if change:
                pos = i
                new_fit = fit

        # Update best individual
        if pos is not None:
            self.__gen = off_gen[pos][np.newaxis]

            if not sames[pos]:
                self.__act = off_act[pos][np.newaxis]
                self.__fit = off_fit[pos][np.newaxis]

        return better, equal, worse, same

    def run (self, merge = 0, on_change = False, log = fh.log):
        """ Run 1 generation """

        self.__generation += 1

        start_time = time.time()
        bet, equ, wor, same = self.one_lambda()
        rtime = time.time() - start_time

        # Save generation data
        self.store_data(bet, equ, wor, same, rtime)

        # Print debug message
        back = merge == 0 or (self.generation - 1) % merge != 0

        if on_change and back:
            back = self.__best_gener[-1] != (self.generation - 1)

        if back:
            print("\033[F\033[K", end = "", file = log)

        self.print_debug(log)

    def run_until (self, stop, merge = 0, on_change = False, log = fh.log):
        """ Run until stop criteria is met """
        merge = max(merge, 0)

        while not stop(self):
            self.run(merge = merge, on_change = on_change, log = log)

    def to_integer (self, rea, mul = None, add = None):
        """ Convert a floating point individual to integer """

        mul = self.integer_mul if mul is None else mul
        add = self.integer_add if add is None else add

        gen = np.empty(rea.shape, dgen)
        np.multiply(rea, mul, casting = "unsafe", out = gen)

        return np.add(gen, add, out = gen)

    def break_genotypes (self, gens):
        """ Break genotypes into functions, arguments, and outputs """

        shape = gens.shape[ : -1 ]
        nod = gens[ ... , : -self.no ].reshape(*shape, -1, self.arity + 1)
        out = gens[ ... , -self.no : ]

        fun = nod[ ... , 0 ]
        arg = nod[ ... , 1 : ]

        return fun, arg, out

    @property
    def find_active_loop (self):
        """ Build a numba compiled loop to find active nodes """

        if self.__p_find_active_loop is None:
            def __find_active_loop (active, active_nod, nod):
                """ Compiled loop to find active nodes """

                for i in range(active.shape[0]):
                    old_found = 0
                    nodi = nod[i]
                    activei = active[i]
                    active_nodi = active_nod[i]

                    # Iterate while there are changes
                    while True:
                        new_active = nodi[active_nodi].reshape(-1)
                        found = new_active.shape[0]

                        if found == old_found:
                            break

                        activei[new_active] = True
                        old_found = found

            # Compile
            self.__p_find_active_loop = nb.njit([
                nb.void(nb.b1[ :, : ], nb.b1[ :, : ], nb.i8[ :, :, : ])
            ], nogil = True)(__find_active_loop)

        return self.__p_find_active_loop

    def find_active (self, gens):
        """ Find active nodes on an individual """

        active = np.zeros(( gens.shape[0], self.full_size ), np.bool8)
        active[ ... , -self.no : ] = True
        active_nod = active[ ... , self.ni : -self.no ]

        # Set outputs as active
        lines = np.arange(gens.shape[0]).reshape(-1, 1)
        active[ lines , gens[ ... , -self.no : ]] = True

        shape = gens.shape[0]
        nod = gens[ ... , : -self.no ].reshape(shape, -1, self.arity + 1)

        fun = nod[ ... , 0 ]
        nod = nod[ ... , 1 : ].copy()

        nod[self.famap[fun]] = self.full_size - 1

        # Propagate active status to output's parents
        self.find_active_loop(active, active_nod, nod)

        return active

    @property
    def simulate_loop (self):
        """ Build a numba compiled loop to simulate circuits """

        if self.__p_simulate_loop is None:
            arities = self.arities
            feval = self.feval
            arr_nodes = self.arr_nodes
            rel_nodes = self.arr_size
            no = self.no

            def __simulate_loop (acts_nod, fun, arg, out, sim):
                """ Compiled loop to simulate circuits """
                for i in range(arg.shape[0]):
                    acts_nodi = acts_nod[i]
                    funi = fun[i]
                    argi = arg[i]
                    simi = sim[i]
                    outi = out[i]

                    for npos in rel_nodes[acts_nodi]:
                        fpos = funi[npos]
                        fari = arities[fpos]
                        farg = argi[npos][ : fari]
                        feval(fpos, simi, farg, arr_nodes[npos])

                    for j in range(-1, -no - 1, -1):
                        simi[j] = simi[outi[j]]

            # Compile
            self.__p_simulate_loop = nb.njit(nogil = True)(__simulate_loop)

        return self.__p_simulate_loop

    def simulate (self, gens, acts, sim):
        """ Simulate circuits into `sim` """
        fun, arg, out = self.break_genotypes(gens)
        acts_nod = acts[ ... , self.ni : -self.no ]

        self.simulate_loop(acts_nod, fun, arg, out, sim)
        return sim

    @property
    def energy_loop (self):
        """ Build a numba compiled loop to evaluate the Landauer's Limit """

        if self.__p_energy_loop is None:
            rel_nodes = self.arr_size
            arities = self.arities
            fnames = self.fnames

            try:
                sim_bits = np.iinfo(self.inp_data.dtype).bits
            except ValueError:
                sim_bits = 1

            nums = self.inp_data.shape[1]
            to_prob = 1.0 / (nums * sim_bits)

            def __energy_loop (fun, arg, sims, act_nod, sim_nod, ene):
                """ Compiled loop to evaluate the Landauer's Limit """

                # Iterate over circuits
                for i in range(len(ene)):
                    simi = sims[i]
                    nods = rel_nodes[act_nod[i]]

                    # Iterate over active nodes
                    for j in range(len(nods)):
                        npos = nods[j]
                        narg = arg[i][npos]
                        fari = arities[fun[i][npos]]

                        c_inp = np.zeros(1 << fari, np.uint64)
                        c_out = np.zeros(2, np.uint64)

                        # Iterate over positions on simulation
                        for k in range(nums):
                            # Iterate over bits on postion
                            for l in range(sim_bits):
                                comb = 0
                                # Fetch output bit
                                opos = (sim_nod[ i, npos, k ] >> l) & 1

                                # Fetch inputs' bits' combinations
                                for m in range(fari):
                                    comb |= ((simi[narg[m], k] >> l) & 1) << m

                                # Save combinations
                                c_inp[comb] += 1
                                c_out[opos] += 1

                        # Calculate energy using Shannon's Entropy
                        energy = 0.0

                        for v in c_out:
                            if v > 0:
                                prob = v * to_prob
                                energy += prob * np.log2(prob)

                        for v in c_inp:
                            if v > 0:
                                prob = v * to_prob
                                energy -= prob * np.log2(prob)

                        ene[i] += energy

            # Compile
            self.__p_energy_loop = nb.njit(nogil = True)(__energy_loop)

        return self.__p_energy_loop

    def energy (self, gens, acts, sims):
        """ Evaluate the energy from circuits """
        fun, arg, _ = self.break_genotypes(gens)
        ene = np.zeros(gens.shape[ : -1 ] or 1, np.double)
        act_nod = acts[ ... , self.ni : -self.no ]
        sim_nod = sims[ ... , self.ni : -self.no , : ]

        self.energy_loop(fun, arg, sims, act_nod, sim_nod, ene)

        return ene

    def get_fitness (self, gens, acts, sims):
        """ Get invidual's fitness """

        # Find matching outputs
        eqs = sims[ ... , -self.no : , : ] == self.exp_data
        eqs = eqs.all(axis = ( 1 , 2 ))

        # Fill results with infinity
        result = np.full(gens.shape[0], np.inf)

        for i, ( gen, act, sim, eq ) in enumerate(zip(gens, acts, sims, eqs)):
            if eq:
                gen = gen[ np.newaxis ]
                act = act[ np.newaxis ]
                sim = sim[ np.newaxis ]

                # Evaluate energy iff all outputs matches
                result[i] = self.energy(gen, act, sim)

        return result

    def print_debug (self, file = fh.log):
        """ Print debug data """

        best_v = self.__best_curve[-1]
        best = ""

        if np.isscalar(best_v):
            best = "{:.5f}".format(best_v)
        else:
            best = " ".join(map("{:.5f}".format, best_v))

        print("Gen", self.generation, end = ": ", file = file)
        print("Best = {}".format(best), end = ", ", file = file)
        print("Same = {}".format(self.__same[-1]), end = ", ", file = file)
        print("Time = {:.5f}s".format(self.__time), end = ", ", file = file)
        print("Diff = {:.5f}s".format(self.__times[-1]), file = file)

    def mutation (self, gen):
        """ Mutate an indivdual """

        randoms = self.__random.random_sample(self.genotype_size)
        selected = randoms < self.mutate_gene

        child = gen.copy()
        changed = self.__random.random_sample(np.count_nonzero(selected))
        child[selected] = self.to_integer(
            changed, self.integer_mul[selected], self.integer_add[selected]
        )

        return child

    def same_critical (self, gen1, act1, gen2):
        """ Test if two genotypes share same critical section """

        # If outputs are different, they may be different
        if not np.array_equal(gen1[ -self.no : ], gen2[ -self.no : ]):
            return False

        inactive = ~act1[ self.ni : -self.no ]

        nod1 = gen1[ : -self.no ].reshape(-1, self.arity + 1).copy()
        nod2 = gen2[ : -self.no ].reshape(-1, self.arity + 1).copy()

        nod1[inactive] = 0
        nod2[inactive] = 0

        fun1 = nod1[ : , 0 ]
        fun2 = nod2[ : , 0 ]

        # If active functions are different, they may be different
        if not np.array_equal(fun1, fun2):
            return False

        nod1 = nod1[ : , 1 : ]
        nod2 = nod2[ : , 1 : ]

        # Test remaining nodes
        return np.logical_or(nod1 == nod2, self.famap[fun1]).all()

    def plot (self, file):
        """ Plot energy convergence curve """

        if plt is None:
            raise Exception("You need `matplotlib` to plot curves.")

        plt.figure()
        plt.plot(
            [ *self.__best_gener, self.generation ],
            [ *self.__best_curve, self.best_fit ],
            color = "#222222", label = "Energy", lw = 0.5
        )

        sciargs = {
            "style"       : "sci",
            "scilimits"   : ( 0 , 0 ),
            "useMathText" : True
        }

        ax = plt.gca()

        gargs = { "axis": "y", "ls": ":" }

        # Add minor locators
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        # Add grids
        plt.grid(True, which = "minor", lw = 0.1, color = "#CCCCCC", **gargs)
        plt.grid(True, which = "major", lw = 0.5, color = "#AAAAAA", **gargs)

        # Use scientific notiation on x-axis
        plt.ticklabel_format(axis = "x", **sciargs)

        # Use scientific notiation on y-axis
        plt.ticklabel_format(axis = "y", **sciargs)

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()

        plt.savefig(file, dpi = 300, transparent = True)

    def verilog (self, gen, act, module = None):
        """ Convert individual to verilog """

        # Max string size is the maximum possible index,
        # + the type identifier 'i' for inputs, 'n' for nodes, 'o' for outputs,
        # + the '~' for possible inverted values
        max_size = len(str(max(( self.ni, self.nc, self.no )))) + 2
        names = np.zeros(self.full_size, "|U{}".format(max_size))
        exprs = []

        inputs = []
        outputs = []
        wires = []

        fun, nod, out = self.break_genotypes(gen)

        # Create inputs' nodes
        for node in range(0, self.ni):
            names[node] = "i{}".format(node)
            inputs.append(node)

        # Slice active nodes
        act_nod = act[ self.ni : -self.no ]

        # Create logical nodes
        for node in self.arr_nodes[act_nod]:
            npos = node - self.ni
            fpos = fun[npos]
            ffmt = self.ffmts[fpos]

            # NOT gate case
            if self.fnames[fpos] == "NOT":
                names[node] = parser.invert(names[nod[ npos, 0 ]])
                continue

            # Create name
            names[node] = "n{}".format(len(wires))
            wires.append(node)

            # Parse args
            args = nod[ npos , : self.arities[fpos] ]
            exprs.append(( node, ffmt.format(*names[args]) ))

        # Create outputs' nodes
        for node, inp in enumerate(out, self.nout_size):
            names[node] = "o{}".format(node - self.nout_size)
            exprs.append(( node, names[inp] ))
            outputs.append(node)

        names = np.array(names)

        # Parse headers
        ionames = "module {} ({});".format(
            self.__module if not module else module,
            ", ".join(it.chain(names[inputs], names[outputs]))
        )
        inames = "  input {};".format(", ".join(names[inputs]))
        onames = "  output {};".format(", ".join(names[outputs]))

        # Create wrapper
        wrap = tw.TextWrapper(
            subsequent_indent = "    ",
            break_long_words = False,
            break_on_hyphens = False,
            width = 80
        )

        # Yield wrapped file
        yield from wrap.wrap(ionames)
        yield ""

        yield from wrap.wrap(inames)
        yield from wrap.wrap(onames)

        if wires:
            wnames = "  wire {};".format(", ".join(names[wires]))
            yield from wrap.wrap(wnames)

        yield ""

        for node, expr in exprs:
            assign = "  assign {} = {};".format(names[node], expr)
            yield from wrap.wrap(assign)

        yield "endmodule"

    def save_curve (self, fname):
        """ Save convergence curve to file """
        with open(fname, "wb") as file:
            np.savez_compressed(file,
                best = self.__best_curve,
                gene = self.__best_gener,
                final = self.generation
            )

    def save_stats (self, fname):
        """ Save evolution stats to file """
        with open(fname, "wb") as file:
            np.savez_compressed(file,
                same = self.same,
                better = self.better,
                equal = self.equal,
                worse = self.worse,
                times = self.times
            )

    def save_best (self, fname):
        """ Save best individual and its data to file """
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        with open(fname, "wb") as file:
            np.savez_compressed(file,
                best = self.best_gen,
                act = self.best_act,
                fit = self.best_fit
            )

    @property
    def popsize (self):
        """ CGP's lambda """
        return self.__popsize

    @property
    def arr_nodes (self):
        """ Array of nodes' indices """
        if self.__p_arr_nodes is None:
            self.__p_arr_nodes = np.arange(self.ni, self.nout_size)
            self.__p_arr_nodes.setflags(write = False)

        return self.__p_arr_nodes

    @property
    def arr_cols (self):
        """ Array of CGP's columns' indices """
        if self.__p_arr_cols is None:
            self.__p_arr_cols = np.arange(self.nc)
            self.__p_arr_cols.setflags(write = False)

        return self.__p_arr_cols

    @property
    def arr_size (self):
        """ Array of CGP's nodes' indices """
        if self.__p_arr_size is None:
            self.__p_arr_size = np.arange(self.nc)
            self.__p_arr_size.setflags(write = False)

        return self.__p_arr_size

    @property
    def inp_data (self):
        """ CGP's input data """
        return self.__inp_data

    @property
    def exp_data (self):
        """ CGP's expected data, given its input data """
        return self.__exp_data

    @property
    def generation (self):
        """ CGP's generation """
        return self.__generation

    @property
    def best_gen (self):
        """ CGP's fittest genotype """
        return self.__gen[0]

    @property
    def best_act (self):
        """ CGP's fittest genotype's active nodes """
        return self.__act[0]

    @property
    def best_fit (self):
        """ CGP's best fitness """
        return self.__fit[0]

    @property
    def best (self):
        """ CGP's fittest genotype's data """
        return self.best_gen, self.best_act, self.best_fit

    @property
    def same (self):
        """ Same individual, i.e., the ones with same critical section \
            as the parent, per generation """
        return self.__same

    @property
    def better (self):
        """ Individuals fitter than the parent per generation """
        return self.__better

    @property
    def equal (self):
        """ Individuals with same fitness as the parent per generation """
        return self.__equal

    @property
    def worse (self):
        """ Individuals less fitter than the parent per generation """
        return self.__worse

    @property
    def times (self):
        """ Generation times """
        return self.__times

    @property
    def time (self):
        """ Total time """
        return self.__time

    @property
    def arity (self):
        """ Maximum arity """
        return self.__arity

    @property
    def ni (self):
        """ Input count """
        return self.__ni

    @property
    def no (self):
        """ Output count """
        return self.__no

    @property
    def nc (self):
        """ Columns count """
        return self.__nc

    @property
    def nout_size (self):
        """ Size of non-outputs """
        return self.ni + self.nc

    @property
    def full_size (self):
        """ Full circuit size """
        return self.nout_size + self.no

    @property
    def gene_size (self):
        """ Gene size """
        return 1 + self.arity

    @property
    def nodes_size (self):
        """ Genotype size of nodes section """
        return self.nc * self.gene_size

    @property
    def genotype_size (self):
        """ Full genotype size """
        return self.nodes_size + self.no

    @property
    def fnames (self):
        """ Gates' functions' names """
        return self.__fnames

    @property
    def funcs (self):
        """ Gates' functions """
        if self.__p_funcs is None:
            funcs = np.array([ parser.gates[name][0] for name in self.fnames ])
            funcs.setflags(write = False)
            self.__p_funcs = funcs

        return self.__p_funcs

    @property
    def arities (self):
        """ Gates' functions' arities """
        if self.__p_arities is None:
            arits = np.array([ parser.gates[name][1] for name in self.fnames ])
            arits.setflags(write = False)
            self.__p_arities = arits

        return self.__p_arities

    @property
    def ffmts (self):
        """ Gates' functions' formatting strings """
        if self.__p_ffmts is None:
            ffmts = np.array([ parser.gates[name][2] for name in self.fnames ])
            ffmts.setflags(write = False)
            self.__p_ffmts = ffmts

        return self.__p_ffmts

    @property
    def famap (self):
        """ Gates' functions' map to disabled inputs """
        if self.__p_famap is None:
            famap = np.zeros(( len(self.fnames), self.arity ), np.bool8)

            for i, arity in enumerate(self.arities):
                famap[ i, arity : ] = True

            famap.setflags(write = False)
            self.__p_famap = famap

        return self.__p_famap

    @property
    def feval (self):
        """ Compiled function to evaluate a gate's function """

        if self.__p_feval is None:
            # Build function's body
            func_txt = "def __eval_func (fpos, values, args, out):\n"
            fmt = "f{}".format

            for i, ( func, fari ) in enumerate(zip(self.funcs, self.arities)):
                # Conditions and calls
                func_txt += (
                    "    {}if fpos == {}:\n"
                    "        {}({}, values[out])\n"
                ).format(
                    ("el" if i else ""), i, fmt(i), ", ".join(
                    "values[args[{}]]".format(j) for j in range(fari)
                ))

            # 'Compile' the function's code to python
            scope = { fmt(i): func for i, func in enumerate(self.funcs) }
            local = {}
            exec(func_txt, scope, local)

            # Compile the python's function
            self.__p_feval = nb.njit(nogil = True)(local["__eval_func"])

        return self.__p_feval

    @property
    def integer_mul (self):
        """ Multiplier to convert genotype from real to integer """
        if self.__p_integer_mul is None:
            i_mul = np.empty(self.genotype_size, dgen)
            i_mul[ -self.no : ].fill(self.nout_size)

            # Its value represent the difference between the maximum
            # and the minimum possible values for each position on the
            # genotype, hence, when multiplied to the real-valued genotype,
            # it returns a new genotype on the range [ 0, max - min ]
            delta_con = self.ni + self.arr_cols
            delta_con[self.arr_cols >= self.nc] = self.nc
            delta_con = np.repeat(delta_con, self.arity)

            nod = i_mul[ : -self.no ].reshape(-1, self.arity + 1)
            nod[ : , 0 ] = len(self.fnames)
            nod[ : , 1 : ] = delta_con.reshape(-1, self.arity)

            i_mul.setflags(write = False)
            self.__p_integer_mul = i_mul

        return self.__p_integer_mul

    @property
    def integer_add (self):
        """ Adder to convert genotype from real to integer, \
            applied after multiplier """

        if self.__p_integer_add is None:
            i_add = np.zeros(self.genotype_size, dgen)

            # Its values represent the minimum possible integer value
            # on the genotype, hence, its summed to the value
            # as the multiplier puts it on the range [ 0, max - min ]
            min_con = np.zeros(self.arr_cols.shape, dgen)
            np.copyto(min_con, self.ni + self.arr_cols - self.nc,
                      where = self.arr_cols >= self.nc)
            min_con = np.repeat(min_con, self.arity)

            nod = i_add[ : -self.no ].reshape(-1, self.arity + 1)
            nod[ : , 1 : ] = min_con.reshape(-1, self.arity)

            i_add.setflags(write = False)
            self.__p_integer_add = i_add

        return self.__p_integer_add

    @property
    def __sims_data (self):
        """ Simulation data storage location """

        if self.__p___sims_data is None:
            # Allocates store space for the simulation of all population
            sims = np.empty(
                ( self.popsize, self.full_size, self.inp_data.shape[1] ),
                self.inp_data.dtype
            )

            sims[ ... , : self.ni , : ] = self.inp_data
            self.__p___sims_data = sims

        return self.__p___sims_data
