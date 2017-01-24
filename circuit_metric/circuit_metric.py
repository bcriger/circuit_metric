# for compatibility between numeric and
# symbolic division:
from __future__ import division

import sympy as sp
import itertools as it
# commentary: When I want to map and reduce using functions that are
# associated with overloaded operators, I import them from operator
# with names that seem weird a priori. Is there a way to make this 
# clear and concise? 
from operator import mul, add, xor as sym_diff, or_ as union
import numpy as np
import qecc as q
from qecc import Location
from collections import defaultdict
import math
import networkx as nx
import vapory as vp

from sys import version_info
if version_info[0] == 3:
    from . import SCLayoutClass as sc
    from functools import reduce
elif version_info[0] == 2:
    import SCLayoutClass as sc


__all__ = [
            "ALLOWED_NAMES", "boundary_dists", "set_prob",
            "r_event_prob", "prob_odd_events", "dict_to_metric",
            "loc_type", "prep_faults", "meas_faults", "str_faults",
            "prop_circ", "synd_set", "synds_to_changes", "syndromes",
            "model_to_pairs", "css_pairs", "fault_probs", "nq",
            "quantify", "metric_to_nx", "css_metrics", "stack_metrics",
            "weighted_event_graph", "metric_to_matrix", "neg_log_odds",
            "apply_step", "fancy_weights", "bit_flip_metric",
            "fault_list"
        ]

#-----------------------------constants-------------------------------#
ALLOWED_NAMES = Location._CLIFFORD_GATE_KINDS
#---------------------------------------------------------------------#

#--------------------------graph manipulation-------------------------#
def boundary_dists(metric):
    """
    Input: metric; a NetworkX.Graph object, consisting of vertices,
    edges and weights.

    Output: a dictionary mapping vertices of the graph to distances 
    calculated using a multi-source Dijkstra algorithm from all of the
    metric boundary vertices to the vertex in question. 

    Note: The boundary vertices in the metric do not correspond to the 
    boundary vertices on the final graph. To produce boundary vertices 
    for the metric, we clone vertices that have weight-one syndromes 
    caused by first-order (probability p) errors in the model. These
    vertices are never placed in the syndrome graph on which an MWPM is
    calculated. Rather, a vertex is created for every syndrome change, 
    and connected to a new vertex by an edge with weight given by this
    function's output. 
    """
    b_nodes = [v for v in metric.nodes() if 'B' in v]
    dist_iter = nx.multi_source_dijkstra_path_length(metric, b_nodes)
    return list(dist_iter)
#---------------------------------------------------------------------#


#-----------------------probability/statistics------------------------#
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
    Input: prob_set; a set of probabilities corresponding to different
    events.
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


def dict_to_metric(pair_p_dict, order=1, weight='neg_log_odds',
                    wt_bits=None, fmt=None):
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
    
    weight = weight.lower()
    wt_lst = ['fowler', 'neg_log_odds', 'prob']
    if weight not in wt_lst:
        raise NotImplementedError("weight must be one of "
            "{}, {} entered.".format(wt_lst, weight))
    
    # TODO: Implement rounded edge weights and remove this Exception.
    if wt_bits is not None:
        raise NotImplementedError("Rounding edge weights to integers"
                                    " is not yet supported.")
    # TODO: Implement an alternate format and remove this Exception.
    if fmt is not None:
        raise NotImplementedError("Returning in different formats is "
                                    "not yet supported.\nReturn value "
                                    "is a list 3-tuple:(vertices, "
                                    "edges, weights).")
    
    # decide whether to use numeric or symbolic log
    vals = uniques(pair_p_dict.values())
    log = appropriate_log(vals)

    #decide based on what's requested whether to give probabilities, 
    #Fowler -log(p) weights, or DKLP -log( p / ( 1 - p ) ) weights. 
    if weight == 'prob':
        val_f = lambda val: prob_odd_events(val, order=order)
    elif weight == 'fowler':
        val_f = lambda val: -log(prob_odd_events(val, order=order))
    elif weight == 'neg_log_odds':
        val_f = lambda val: neg_log_odds(val, order=order)

    pair_p_dict = {key: val_f(val) for key, val in pair_p_dict.items()}

    vertices = uniques(pair_p_dict.keys())
    edges, weights = map(list, zip(*pair_p_dict.items()))
    
    # Append boundary vertices:
    for edge_dx in range(len(edges)):
        edge = edges[edge_dx]
        if len(edge) == 1:
            # should connect to boundary
            new_vertex = ('B',) + edge[0]
            vertices.append(new_vertex)
            edges[edge_dx] = (edge[0], new_vertex)

    return vertices, edges, weights

def metric_to_matrix(vertices, edges, weights):
    
    n = len(vertices)
    mat_out = np.zeros((n, n), dtype=np.float64)
    
    vs = sorted(vertices)
    for e, w in zip(edges, weights):
        r, c = map(vs.index, e)
        #suspect that matrix should be symmetric
        mat_out[r, c] = w
        mat_out[c, r] = w
    
    return mat_out

#---------------------------------------------------------------------#

#------------------------circuit manipulation-------------------------#
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
    
    :param qecc.Circuit circuit: Subcircuit in which faults are to 
    be considered.

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
    quaec_circs = list(dumb_circ.group_by_time(pad_with_waits=waits))

    return quaec_circs


def synd_set(circ, fault, time, prop_twice=True):
    """
    for a single fault, propagates over a circuit with prep/meas
    locations and returns the syndrome pair as a pair of 3D 
    coordinates.
    """
    cliffs = map(lambda c: c.as_clifford(), prop_circ(circ, waits=True))

    output = [[], []]
    n_bits = nq(circ)  # dumb
    #TODO Make this its own function
    for step, prop in zip(circ[time + 1:], cliffs[time + 1:]):
        # preps eliminate faults
        for idx in [tpl[1] for tpl in loc_type(step, 'P')]:
            fault.op = fault.op[:idx] + 'I' + fault.op[idx + 1:]
        # measurements make syndromes
        output[0].extend(syndromes(step, fault, n_bits))
        # step forward
        fault = prop.conjugate_pauli(fault)

    if prop_twice:
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
        crd_lst.append(layout.map.inv[synd[1]] + (0,))
    
    for synd in synds[0] + synds[1]:
        if (synd in synds[0]) != (synd in synds[1]):
            crd_lst.append(layout.map.inv[synd[1]] + (1,))
    
    return tuple(crd_lst)


def syndromes(step, fault, n_bits):
    """
    Determines which measurement locations in a circuit are activated 
    by a given fault.
    The circuit is a single syndrome extractor.
    """
    synd_lst = []
    
    for loc in step:
        #read pauli type off of location label (ugly)
        if 'M_' in loc[0]:
            ltr = loc[0][-1]
            test_pauli = q.Pauli.from_sparse({loc[1]: ltr}, n_bits)
            if q.com(test_pauli, fault) == 1:
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
        step_dict = {
                    synds_to_changes(layout, synd_set(circ, f, t)): p
                    for f, p in f_ps[t]
                    }
        for key, val in step_dict.items():
            output[key].append(val)

    del output[()] #ignore syndrome-less errors

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

    del pairs[()] #ignore errors with inappropriate syndromes

    return pairs

def fault_list(circ, p, test=False):
    prep = prep_faults(circ)
    meas = meas_faults(circ)
    cnot = str_faults(circ, 'CNOT')
    wait = str_faults(circ, 'I')
    # TODO: H, P, CZ faults (by this point you'll want a new model)

    # It looks like we need to find a better way to do this, but we 
    # don't.
    # It's just a test model, we'll have something much worse IRL.
    out_lst = [[] for elem in prep]
    for dx in range(len(prep)):
        if test:
            out_lst[dx].extend([(f, p, 'p') for f in prep[dx]])
            out_lst[dx].extend([(f, p / 3, 'o') for f in wait[dx]])
            out_lst[dx].extend([(f, p / 15, 'o') for f in cnot[dx]])
            out_lst[dx].extend([(f, p, 'm') for f in meas[dx]])
        else:
            out_lst[dx].extend([(f, p) for f in prep[dx]])
            out_lst[dx].extend([(f, p / 3) for f in wait[dx]])
            out_lst[dx].extend([(f, p / 15) for f in cnot[dx]])
            out_lst[dx].extend([(f, p) for f in meas[dx]])
        # post-process: determine if the same error happens twice in the
        # same timestep. If so, adjust the probability of that Pauli so 
        # that it's the probability of that Pauli happening an odd number
        # of times.
        # This happens with measurement errors, which are in the timestep
        # *before the measurement* instead of after (like every other 
        # gate).
        key_set = list(set([tpl[0] for tpl in out_lst[dx]]))
        for key in key_set:
            if quantify(out_lst[dx], lambda tpl: tpl[0] == key) > 1:
                repeat_ps = [tpl[1] for tpl in out_lst[dx] if tpl[0] == key]
                unique_p = prob_odd_events(repeat_ps, len(repeat_ps))
                out_lst[dx] = [elm for elm in out_lst[dx] if elm[0] != key]
                out_lst[dx].append((key, unique_p))
        
    return out_lst, circ
#---------------------------------------------------------------------#

#-----------------------surface code specifics------------------------#
def fault_probs(distance, p=None, test=False):
    """
    Returns a list which is as long as the syndrome extractor. Each 
    entry contains the Paulis which may occur immediately after that 
    timestep, and a symbolic/numeric probability based on a hardcoded
    symmetric error model.

    if testing, assigns a string to the pair, so you can tell which
    faults do what.
    """

    layout = sc.SCLayout(distance)
    circ = layout.extractor()
    p = p if p else sp.Symbol('p')

    out_lst, circ = fault_list(circ, p, test)
    return out_lst, circ, layout
#---------------------------------------------------------------------#

#-----------------------convenience functions-------------------------#
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


def quantify(iterable, pred=bool):
    """
    Counts how many times a predicate is true for elements in an 
    iterable
    """
    return sum(it.imap(pred, iterable))


def metric_to_nx(vertices, edges, weights):
    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    graph.add_weighted_edges_from([
                                    e + (w,)
                                    for e, w in zip(edges, weights)
                                ])
    return graph


v_shft = lambda v, t: v[:-1] + (v[-1] + t,)
v_shft.__doc__ = "Shifts the last co-ordinate of a tuple v by t." 

def neg_log_odds(p_set, order):
    p_odd = prob_odd_events(p_set, order=order)
    log = appropriate_log(p_set)
    return -log( p_odd / ( 1. - p_odd ))

def appropriate_log(vals):
    if any([isinstance(val, sp.Symbol) for val in vals]):
        return sp.log
    else:
        return np.log

def fancy_weights(prob_mat, subtract_diag=False, distance=None):
    """
    From Tom O'Brien:
    If you want to evaluate the probability of getting from vertex A to
    vertex B with a weird SAW, you can sum powers of a weighted
    adjacency matrix for the vertices. FALSE. You have to subtract off the non-SA walks. This is a TODO.
    If you take the limit as step length goes to infinity, this sum
    converges to: 
    -(P - I)^{-1} - I
    as long as |P| < 1. 
    Since all weights p are 0 < p < 1, we're probably good. 
    """
    nlo = np.vectorize(lambda p: -math.log( p / (1. - p) ))
    idnt = np.identity(prob_mat.shape[0])
    if subtract_diag:
        if distance is None:
            raise ValueError("if you set subtract_diag to True, you "
                                "have to input a distance.")
        p_sum_mat = prob_mat.copy()
        for step_dx in range(2 * distance):
            temp_mat = p_sum_mat * prob_mat
            temp_mat -= np.diagonal(np.diagonal(temp_mat))
            p_sum_mat += temp_mat
    else:
        p_sum_mat = np.linalg.inv(idnt - prob_mat) - idnt 
    return nlo(p_sum_mat)

def apply_step(step, pauli):
    """
    Silently applies a list of gates 'timestep' to a sparse_pauli.Pauli
    'pauli' (pauli).
    step is meant to be an iterable of gates of the form
    ('name', qubit0, qubit1(optional)), where name can be one of:
    I, P_X, P_Z, M_X, M_Z, CNOT
    This is meant to expand as we change gatesets to Y90/CPHASE.
    """
    #FIXME assumes correct input
    qubits = reduce(union, map(set, [tpl[1:] for tpl in step]))
    ids, x_ps, x_ms, z_ps, z_ms = [
        [tpl[1] for tpl in step if tpl[0] == name]
        for name in ['I', 'P_X', 'M_X', 'P_Z', 'M_Z']
        ]
    cnots = [tpl[1:] for tpl in step if tpl[0] == 'CNOT']
    # check if sets don't overlap by taking difference between sym_diff
    # and union
    cnot_qs = sum(cnots, ())

    #lazy check, doesn't work if three gates are on the same qubit. 
    #FIXME
    test_qs = reduce(sym_diff,
                map(set,
                    [ids, x_ps, z_ps, x_ms, z_ms, cnot_qs]
                    )
                )
    
    if test_qs != qubits:
        raise ValueError("qubit sets must not intersect.")

    #method-based implementation of gates kind of sucks. 
    pauli.cnot(cnots)
    pauli.prep(x_ps + z_ps)
    #these can always be in the wrong order
    z_synds = pauli.meas(z_ms, basis='Z')
    x_synds = pauli.meas(x_ms, basis='X')
    
    return ({'X': x_synds, 'Z': z_synds}, pauli)

#---------------------------------------------------------------------#

#------------------------user-level functions-------------------------#
def css_metrics(model, circ, layout, weight='neg_log_odds'):
    """
    Just a little something to make generating MWPM metrics a little 
    easier. 
    Note: `circ` is usually the result of a Layout method. 
    """
    pairs = model_to_pairs(model, circ, layout)
    x_dict, z_dict = map(lambda _: css_pairs(pairs, layout, _), 'xz')
    x_metric = dict_to_metric(x_dict, weight=weight)
    z_metric = dict_to_metric(z_dict, weight=weight)
    return x_metric, z_metric


def stack_metrics(metric_lst):
    """
    Input: a list or iterable of (vertices, edges, weights) tuples,
    which correspond to measurement layers (2 consecutive rounds of
    measurement, where vertex time co-ordinates are 0/1).

    Output: a large (vertices, edges, weights) tuple, containing 
    information on all rounds of measurement. 
    
    Notes:
        Use this function before converting to NetworkX.
        This function allows us to model time-dependent errors.
    """
    vs, es, ws = [], [], []

    for m_dx, metric in enumerate(metric_lst):
        vs.extend([ v_shft(v, m_dx) for v in metric[0]])
        es.extend([ 
                    map(lambda v: v_shft(v, m_dx), e)
                    for e in metric[1]
                    ])
        ws.extend(metric[2])
    return (list(set(vs)), es, ws)

def weighted_event_graph(event_graph, metric):
    """
    Adds weights to a graph of 'detection events', by passing each pair
    of vertices to a single-source Dijkstra algorithm. 
    While this may not be necessary for the highly-symmetric test
    metrics produced in theoretical studies, there are no guarantees on
    whether hypotenuse edges are lower in weight than the sum of the 
    weights of the associated perpendicular edges.
    Also, there may be 'hot spots' (short chains of times/locations
    where errors are very likely) that justify not taking a straight
    path from point A to point B. 

    Warning: if you put in a weighted graph, this will overwrite your
    weights.  
    """
    for edge in event_graph.edges():
        event_graph[edge[0]][edge[1]]['weight'] = \
            nx.dijkstra_path_length(metric, edge[0], edge[1])

    return event_graph

def visualize(metric_stack, flnm):
    """
    Uses POVRay (through vapory) to produce a Fowler-like nest diagram. 
    """
    #FIXME Finish
    raise NotImplementedError("This is not done yet.")
    verts, edges, weights = metric_stack

    #position camera beyond largest x/y/z co-ord.
    camera_pos = map(max, zip(*verts))

    camera = Camera( 'location', [0,2,-3], 'look_at', [0,1,2] )
    light = LightSource( [2,4,-3], 'color', [1,1,1] )
    sphere = Sphere( [0,1,2], 2, Texture( Pigment( 'color', [1,0,1] )))

    scene = Scene( camera, objects = node_lst + edge_lst)
    scene.render(flnm, width=2000, height=2000)

def bit_flip_metric(d, p, ltr='x'):
    """
    I have a sneaking suspicion that using the new metric from Tom is
    going to fix my low threshold. 
    This has something to do with boundary conditions; only when you're
    not on the torus do the fancy weights have a 'tie-breaking' effect. 
    The idea here is to put all the boundary vertices in the metric, 
    so that the effective length to the closest vertex can be found. 
    Some of these ideas are not well-justified, but I'll worry about
    that if I keep having low thresholds. 
    ltr is the type of ERROR you're trying to correct.

    I output enough info to make a nice function that takes you from
    co-ordinates to weights like Sim2D.pair_dist does. 
    """
    ltr = ltr.lower()
    layout = sc.SCLayout(d)
    if ltr == 'x':
        vertices = layout.z_ancs() + layout.boundary_points('x')
    elif ltr == 'z':
        vertices = layout.x_ancs() + layout.boundary_points('z')

    vertices = sorted(vertices)
    g = nx.Graph()
    g.add_nodes_from(vertices)
    shfts = [(2,2), (2,-2), (-2, 2), (-2, -2)]
    for vertex in vertices:
        for shft in shfts:
            other_vertex = (vertex[0] + shft[0], vertex[1] + shft[1])
            if other_vertex in vertices:
                g.add_edge(vertex, other_vertex)

    # That's the graph done. Now let's get the adjacency matrix and
    # fancy it.
    adj_mat = nx.adjacency_matrix(g, nodelist=vertices).todense()
    wts = fancy_weights(p * adj_mat)
    return vertices, wts / np.amin(wts)

#---------------------------------------------------------------------#
