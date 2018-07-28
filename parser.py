import re
import time
import string
import itertools as it

import numpy as np
import numba as nb


class ParserException (Exception):
    """ Class to represent parser's exceptions """

@nb.vectorize(nopython = True, cache = True)
def np_bitwise_maj (x, y, z):
    """ Bitwise MAJ function """
    return (x & y) | (x & z) | (y & z)

maj_fmt = "({0} & {1}) | ({0} & {2}) | ({1} & {2})"

gates = {
    "AND"  : ( np.bitwise_and , 2 , "{} & {}" ),
    "OR"   : ( np.bitwise_or  , 2 , "{} | {}" ),
    "MAJ"  : ( np_bitwise_maj , 3 , maj_fmt   ),
    "NOT"  : ( np.bitwise_not , 1 , "~{}"     ),
}

# Pre-compiled REGEXPs
remove_newline_comment = re.compile(r"//.*")
remove_two_not = re.compile(r"^(?:~~)+")
single_spaces = re.compile(r"\s+")
norm_spaces = re.compile(r"\s*([{},;()&|=])\s*")
binary_input = re.compile(r"^~?(?:1'b)?([01])$")
gates_split = re.compile(r"[&|]")
parse_file = re.compile(r"^module ([^\s]+) \((?:[^;]*)\) ; "
                        r"input ([^;]*) ; "
                        r"output ([^;]*) ; "
                        r"wire (?:[^;]*) ; "
                        r"((?:.* ; )*?)"
                        r"endmodule$")

def plain (gate):
    """ Returns non-inverted gate name """
    return gate.lstrip("~")

def invert (gate):
    """ Inverts the gate name """
    return plain(gate) if gate[0] == "~" else "~{}".format(gate)

def split_file (fname):
    """ Splits file into pieces """

    data = ""

    with open(fname, "r") as f:
        data = f.read()

    info = parse_file.match(single_spaces.sub(r" ",
        norm_spaces.sub(r" \1 ", remove_newline_comment.sub("", data))
    ).strip())

    if info is None:
        raise Exception("Could not match file.")

    # Break output into metadata
    return (
        info.group(1).strip(),
        list(map(str.strip, info.group(2).split(","))),
        list(map(str.strip, info.group(3).split(","))),
        list(filter(bool, map(str.strip, info.group(4).split(";"))))
    )

def sanitize_input (inp):
    """ Sanitizes inputs' names by removing \
        double inversors and finding fixed inputs (0 or 1) """

    inp = remove_two_not.sub("", inp)

    if binary_input.match(inp):
        raise ParserException("Fixed inputs are not supported.")

    return inp

def parse_args (rop, out):
    """ Parse arguments """

    args = []
    unique = set()

    strip = "();{}".format(string.whitespace)

    for op in gates_split.split(rop):
        op = sanitize_input(op.strip(strip))

        if op == "" or op in unique:
            continue

        unique.add(op)
        args.append(op)

    return args

def parse_expression (expr):
    """ Parse expression """

    lop, rop = expr[ 7 : ].split("=", 1)
    out = lop.strip()

    # Count ands and ors
    and_count = rop.count("&")
    or_count = rop.count("|")

    args = parse_args(rop, out)

    # Tries to figure out the gate
    if len(args) == 1:
        gate = "I"

    elif len(args) == 2:
        if and_count == 1:
            gate = "AND"
        elif or_count == 1:
            gate = "OR"

    elif len(args) == 3:
        if (and_count == 2 or and_count == 3) and or_count == 2:
            gate = "MAJ"

    if gate is None:
        raise ParserException("Unknown operator at `{}`.".format(expr))

    return out, gate, args

def topological_sort (inputs, parents, children):
    """ Make a topological sort of the circuit """

    # Start on the inputs
    order = inputs.copy()
    pos = 0

    while pos < len(order):
        node = order[pos]

        # Iterate over children
        for child in children.get(node, []):
            parents[child] -= 1

            if parents[child] == 0:
                # Add to order if all parents were processed
                order.append(child)

        pos += 1

    return order

def find_id (node, node_map, ids, circuit):
    """ Find node id or insert it if inverted """

    node = node_map.get(node, node)

    if node not in ids:
        # Create inversor if needed
        if node[0] == "~":
            ids[node] = len(ids)
            circuit.append([ "NOT", ids[plain(node)] ])

    # Could not find a valid argment
    if node not in ids:
        raise ParserException("Node `{}` undefined.".format(node))

    return ids[node]


def build_circuit (order, inputs, gates, outputs):
    """ Build circuit from pre-processed pieces """

    # Assign IDs
    ids = { inp : i for i, inp in enumerate(inputs) }
    # Map to remove wires and indirect double inversors, e.g., a = ~b; c = ~a
    node_map = {}
    # Build circuit
    circuit = []

    # Iterate over topology, ignoring inputs
    for pos in range(len(inputs), len(order)):
        node = order[pos]

        # Node does not have an expression
        if node not in gates:
            continue

        gate, args = gates[node]

        if gate == "I":
            # Parse wires and inversors
            arg = args[0]
            iarg = invert(arg)

            # Add node and its inverted value to map
            node_map[node] = node_map.get(arg, arg)
            node_map[invert(node)] = node_map.get(iarg, iarg)
            continue

        # Fetch arguments
        arg_ids = [ find_id(arg, node_map, ids, circuit) for arg in args ]

        ids[node] = len(ids)
        circuit.append([ gate, *arg_ids ])

    # Convert outputs to ids
    outputs = [ find_id(out, node_map, ids, circuit) for out in outputs ]

    return circuit, outputs

def convert (fname):
    """ Convert verilog to a format closer to CGP's genotype """

    # Split file data
    module, inputs, outputs, exprs = split_file(fname)

    parents = {}
    children = {}
    gates = {}

    # Parse expressions
    for out, gate, args in map(parse_expression, exprs):
        parents[out] = len(args)
        gates[out] = ( gate, args )

        for arg in map(plain, args):
            children.setdefault(arg, []).append(out)

    order = topological_sort(inputs, parents, children)
    circuit, outputs = build_circuit(order, inputs, gates, outputs)

    return module, inputs, circuit, outputs
