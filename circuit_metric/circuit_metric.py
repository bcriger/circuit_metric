# for compatibility between numeric and
# symbolic division:
from __future__ import division

import sympy as sp
import itertools as it
from operator import mul, add
import numpy as np
import qecc as q
from qecc import Location
from collections import defaultdict
import networkx as nx
import vapory as vp

from sys import version_info
if version_info[0] == 3:
    #from . import SCLayoutClass as sc
    from . import SCLayoutClass as sc
elif version_info[0] == 2:
    import SCLayoutClass as sc


__all__ = [
            "ALLOWED_NAMES", "boundary_dists", "set_prob",
            "r_event_prob", "prob_odd_events", "dict_to_metric",
            "loc_type", "prep_faults", "meas_faults", "str_faults",
            "prop_circ", "synd_set", "synds_to_changes", "syndromes",
            "model_to_pairs", "css_pairs", "fault_probs", "nq",
            "quantify", "metric_to_nx", "css_metrics", "stack_metrics",
            "weighted_event_graph", "metric_to_matrix", "neg_log_odds"
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
    
    :param qecc.Circuit circuit: Subcircuit to in which faults are to 
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

def fancy_weights(prob_mat):
    """
    From Tom O'Brien:
    If you want to evaluate the probability of getting from vertex A to
    vertex B with a weird SAW, you can sum powers of a weighted
    adjacency matrix for the vertices. 
    If you take the limit as step length goes to infinity, this sum
    converges to: 
    -(P - I)^{-1} - I
    as long as |P| < 1. 
    Since all weights p are 0 < p < 1, we're probably good. 
    """
    nlo = np.vectorize(lambda p: -np.log( p / (1. - p) ))
    idnt = np.identity(prob_mat.shape)
    return nlo(-np.linalg.inv(test_mat - idnt) - idnt)


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
#---------------------------------------------------------------------#
