# for compatibility between numeric and
# symbolic division:
from __future__ import division

import sympy as sp
import itertools as it
from operator import mul, add
import numpy as np
import qecc as q
import SCLayoutClass as sc
from qecc import Location
from collections import defaultdict
import TestClass as tc

# -----------------------------constants-------------------------------#
ALLOWED_NAMES = Location._CLIFFORD_GATE_KINDS

# ---------------------------------------------------------------------#

# -----------------------probability/statistics------------------------#


def set_prob(ps, ys):
    """
    Input: ps; a list of probabilities of certain events occurring
    Input: ys; indices indicating which of the events occurs ('yesses')
    Output: the probability that this event set occurs
    """
    dxs = range(len(ps))
    return product(ps[dx] if dx in ys else 1 - ps[dx] for dx in dxs)


def r_event_prob(prob_set, r=1):
    """
    Input: prob_set; a set of probabilities corresponding to different events.
    Input: r; a number of events which occur (default 1). 
    Output: the probability of exactly r events occurring.
    """
    idxs = range(len(prob_set))
    idx_sets = it.combinations(idxs, r=r)

    return sum(set_prob(prob_set, idx_set) for idx_set in idx_sets)


def prob_odd_events(prob_set, order=1):
    """
    Returns the probability that an odd number of events from a given 
    set has occurred, to a given order in a Taylor expansion. This 
    expansion is evaluated by noting that the r'th-order term 
    corresponds to the cases in which r events occur. 
    Input: prob_set; a set of probabilities (numeric or symbolic)
    Input: order; the highest-order term in the expansion
    Output: a probability sum, type determined by input.
    """
    rs = range(1, order + 1, 2)
    return sum(r_event_prob(prob_set, r) for r in rs)


def dict_to_metric(pair_p_dict, order=1, wt_bits=None, fmt=None):
    """
    The output of the error propagation and combinatronics is a
    dictionary whose keys are syndrome pairs and whose values are 
    sets of probabilities. This is enough information to construct the
    metric, but we do that here, because of a few issues that may 
    arise.

    First, you might have to restrict to uint edge weights with some 
    small size. Double-precision floats will be returned by default, 
    unless wt_bits is set, which gives the number of bits to which
    the weights are rounded.

    Second, you might want the complete graph in a certain format; 
    weighted adjacency matrix, NetworkX, array form for input into 
    Blossom, etcetera. These formats will all be associated with the
    optional argument 'fmt'. 

    Optional arguments are included in the signature as a reminder, 
    but these features are not yet implemented.
    """
    # TODO: Implement rounded edge weights and remove this Exception.
    if wt_bits is not None:
        raise NotImplementedError("Rounding edge weights to integers is not yet supported.")
    # TODO: Implement an alternate format and remove this Exception.
    if fmt is not None:
        raise NotImplementedError("Returning in different formats is not yet supported.\n"
                                  "Return value is a list 3-tuple:(vertices, edges, weights).")
    
    # decide whether to use numeric or symbolic log
    if any([isinstance(val, sp.Symbol) for val in uniques(pair_p_dict.values())]):
        log = sp.log
    else:
        log = np.log

    pair_p_dict = {key: -log(prob_odd_events(val, order=order)) for key, val in pair_p_dict.items()}

    vertices = uniques(pair_p_dict.keys())
    edges, weights = map(list, zip(*pair_p_dict.items()))
    
    # Append virtual vertices when you make the graph, not when you make
    # the metric:
    '''
    #append virtual vertices, assume they are tuples
    virtual_vertices = [vertex + ('b',) for vertex in vertices]
    vertices.extend(virtual_vertices)
    #weight-0 edges between all virtual vertices
    weight_0_edges = list(it.combinations(virtual_vertices, r=2))
    edges.extend(weight_0_edges)
    weights.extend([0 for _ in range(len(weight_0_edges))])
    '''

    return vertices, edges, weights

# ---------------------------------------------------------------------#

# ------------------------circuit manipulation-------------------------#


def loc_type(timestep, string):
    """
    returns all locations in a timestep whose identifiers contain a 
    certain string.
    """

    return filter(lambda tpl: string in tpl[0], timestep)


def prep_faults(circ):
    """
    Expands on the fault lists provided by QuaEC by producing faults 
    in the basis perpendicular to the eigenbasis of the prepared state.
    """
    n_bits = nq(circ)
    return [[q.Pauli.from_sparse({tpl[1]:prep_fault[tpl[0]]}, nq=n_bits)
            for tpl in loc_type(step, 'P_')]
            for step in circ]


def meas_faults(circ):
    """
    Expands on the fault lists provided by QuaEC by producing faults 
    in the basis perpendicular to the eigenbasis of the measurement 
    operator. These faults occur in the round before the measurement
    with which they are associated.
    """
    n_bits = nq(circ)
    return [[q.Pauli.from_sparse({tpl[1]: meas_fault[tpl[0]]}, nq=n_bits)
            for tpl in loc_type(step, 'M_')]
            for step in circ][1:] + [[]]


def str_faults(circ, gt_str):
    """
    horrendous copypaste from quaec

    Takes a sub-circuit which has been padded with waits, and returns an
    iterator onto Paulis which may occur as faults after this sub-circuit.
    
    :param qecc.Circuit circuit: Subcircuit to in which faults are to be considered.

    """
    big_lst = [[] for _ in range(len(circ))]
    # god forgive me
    nq = prop_circ(circ, waits=True)[0].nq
    
    for dx, step in enumerate(prop_circ(circ)):
        big_lst[dx] = filter(lambda p: p.wt != 0,
                             list(it.chain.from_iterable(
                              q.restricted_pauli_group(loc.qubits, nq)
                              for loc in step if loc.kind == gt_str
                                )))
    
    return big_lst


def prop_circ(circ_lst, waits=False):
    """
    QuaEC circuit through which errors will be propagated.
    We remove prep and measurement locations for QuaEC compatibility. 
    """
    # circ_lst += circ_lst
    prop_lst = map(lambda lst: filter(is_allowed, lst), circ_lst)
    dumb_circ = q.Circuit(*sum(prop_lst, []))
    #print 'dumb_circ', dumb_circ
    quaec_circs = list(dumb_circ.group_by_time(pad_with_waits=waits))

    return quaec_circs


def synd_set(circ, fault, time):
    """
    for a single fault, propagates over a circuit with prep/meas
    locations and returns the syndrome pair as a pair of 3D 
    coordinates.
    """
    cliffs = map(lambda c: c.as_clifford(), prop_circ(circ, waits=True))

    output = [[], []]
    n_bits = nq(circ)  # dumb

    for step, prop in zip(circ[time + 1:], cliffs[time + 1:]):
        # preps eliminate faults
        for idx in [tpl[1] for tpl in loc_type(step, 'P')]:
            fault.op = fault.op[:idx] + 'I' + fault.op[idx + 1:]
        # measurements make syndromes
        output[0].extend(syndromes(step, fault, n_bits))
        # step forward
        fault = prop.conjugate_pauli(fault)
    for step, prop in zip(circ, cliffs):
        # preps eliminate faults
        for idx in [tpl[1] for tpl in loc_type(step, 'P')]:
            fault.op = fault.op[:idx] + 'I' + fault.op[idx + 1:]
        # measurements make syndromes
        output[1].extend(syndromes(step, fault, n_bits))
        # step forward
        fault = prop.conjugate_pauli(fault)

    return output


def synds_to_changes(layout, synds):
    """
    given two cycles' worth of syndrome information for a certain 
    fault, produces vertex sets with co-ordinates given by the layout.
    """

    crd_lst = []

    for synd in synds[0]:
        crd_lst.append(layout.map[:synd[1]] + (0,))
    
    for synd in synds[0] + synds[1]:
        if (synd in synds[0]) != (synd in synds[1]):
            crd_lst.append(layout.map[:synd[1]] + (1,))
    
    return tuple(crd_lst)


def syndromes(step, fault, n_bits):
    """
    Determines which measurement locations in a circuit are activated 
    by a given fault.
    The circuit is a single syndrome extractor
    """
    synd_lst = []
    
    for loc in step:
        if 'M_' in loc[0]:
            ltr = loc[0][-1]
            test_pauli = q.Pauli.from_sparse({loc[1]: ltr}, n_bits)
            if q.com(test_pauli, fault):
                synd_lst.append(loc)

    return synd_lst


def model_to_pairs(f_ps, circ, layout):
    """
    Takes output from a function like fault_probs and translates the 
    faults into syndromes, using the fact that the list is 
    time-ordered.
    """
    output = defaultdict(list)

    for t in range(len(f_ps)):
        step_dict = {synds_to_changes(layout, synd_set(circ, tpl[0], t)): tpl[1] for tpl in f_ps[t]}
        for key, val in step_dict.items():
            output[key].append(val)

    return output


def css_pairs(synds, layout, synd_tp):
    """
    Assumes that errors are uncorrelated to perform CSS decoding. 
    Separates syndromes into x and z type using the x and z ancillas
    of the layout.
    """
    if synd_tp.lower() == 'x':
        ancs = layout.x_ancs()
    elif synd_tp.lower() == 'z':
        ancs = layout.z_ancs()
    
    pairs = defaultdict(list)

    for key, val in synds.items():
        pairs[tuple(filter(lambda j: j[:-1] in ancs, key))].extend(val)

    return pairs

# ---------------------------------------------------------------------#

# -----------------------surface code specifics------------------------#


def fault_probs(distance, test=False):
    """
    Returns a list which is as long as the syndrome extractor. Each 
    entry contains the Paulis which may occur immediately after that 
    timestep, and a symbolic probability based on a hardcoded symmetric
    error model.

    if testing, assigns an integer tuple to the probability, so you can
    tell which faults do what
    """
    layout = sc.SCLayout(distance)
    circ = layout.extractor()
    p = sp.Symbol('p')

    prep = prep_faults(circ)
    meas = meas_faults(circ)
    cnot = str_faults(circ, 'CNOT')
    wait = str_faults(circ, 'I')
    # TODO: H, P, CZ faults (by this point you'll want a new model)

    out_lst = [[] for elem in prep]
    for dx in range(len(prep)):
        if test:
            out_lst[dx].extend([(f, p, 'p') for f in prep[dx]])
            out_lst[dx].extend([(f, p, 'm') for f in meas[dx]])
            out_lst[dx].extend([(f, p / 3, 'o') for f in wait[dx]])
            out_lst[dx].extend([(f, p / 15, 'o') for f in cnot[dx]])
        else:
            out_lst[dx].extend([(f, p) for f in prep[dx]])
            out_lst[dx].extend([(f, p) for f in meas[dx]])
            out_lst[dx].extend([(f, p / 3) for f in wait[dx]])
            out_lst[dx].extend([(f, p / 15) for f in cnot[dx]])

    return out_lst, circ, layout

# ---------------------------------------------------------------------#

# -----------------------convenience functions-------------------------#

is_allowed = lambda tpl: tpl[0] in ALLOWED_NAMES
is_allowed.__doc__ = """tests whether the zeroth element of a tuple is
 in the allowed list of gate names from qecc.Location""" 
 
uniques = lambda x: list(set(reduce(add, x)))
uniques.__doc__ = "returns unique elements from an iterator"

product = lambda num_iter: reduce(mul, num_iter)

prep_fault = {'P_X': 'Z', 'P_Z': 'X'}
meas_fault = {'M_X': 'Z', 'M_Z': 'X'}

qubits = lambda tpls: list(set(reduce(add, [t[1:] for t in tpls])))


def nq(circ):
    """
    This is a dirty hack to restore some of QuaEC's circuit 
    functionality to circuits with prep and measurement locations 
    """
    all_bits = list(set(reduce(add, [qubits(_) for _ in circ])))
    return len(all_bits)

# ---------------------------------------------------------------------#


def fault_probs1(test=False):
    """
    Returns a list which is as long as the syndrome extractor. Each
    entry contains the Paulis which may occur immediately after that
    timestep, and a symbolic probability based on a hardcoded symmetric
    error model.

    if testing, assigns an integer tuple to the probability, so you can
    tell which faults do what
    """
    layout = tc.TestClass()
    circ = layout.extractor()
    p = sp.Symbol('p')

    prep = prep_faults(circ)
    meas = meas_faults(circ)
    cnot = str_faults(circ, 'CNOT')
    wait = str_faults(circ, 'I')
    # TODO: H, P, CZ faults (by this point you'll want a new model)

    out_lst = [[] for elem in prep]
    for dx in range(len(prep)):
        if test:
            out_lst[dx].extend([(f, p, 'p') for f in prep[dx]])
            out_lst[dx].extend([(f, p, 'm') for f in meas[dx]])
            out_lst[dx].extend([(f, p / 3, 'o') for f in wait[dx]])
            out_lst[dx].extend([(f, p / 15, 'o') for f in cnot[dx]])
        else:
            out_lst[dx].extend([(f, p) for f in prep[dx]])
            out_lst[dx].extend([(f, p) for f in meas[dx]])
            out_lst[dx].extend([(f, p / 3) for f in wait[dx]])
            out_lst[dx].extend([(f, p / 15) for f in cnot[dx]])

    return out_lst, circ, layout