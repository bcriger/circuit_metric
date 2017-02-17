# The import syntax changes slightly between python 2 and 3, so we
# need to detect which version is being used:
from sys import version_info
if version_info[0] == 3:
    PY3 = True
    from importlib import reload
    from functools import reduce
elif version_info[0] == 2:
    PY3 = False
else:
    raise EnvironmentError("sys.version_info refers to a version of "
        "Python neither 2 nor 3. This is not permitted. "
        "sys.version_info = {}".format(version_info))

import bidict as bd
import itertools as it
from .layout_utils import *
from math import copysign

try:
    import matplotlib.pyplot as plt
except RuntimeError:
    pass #hope you don't want to draw

import networkx as nx
import numpy as np
from operator import add
import sparse_pauli as sp

#------------------------------constants------------------------------#

SHIFTS_README = """(dx, dy) so that, given an ancilla co-ordinate
                   (x, y), there will be data qubits at
                   (x + dx, y + dy)."""
SHIFTS = {
            'N': ((-1, 1), (1, 1)),
            'E': ((1, 1), (1, -1)),
            'W': ((-1, 1), (-1, -1)),
            'S': ((1, -1), (-1, -1))
        }
SHIFTS['A'] = SHIFTS['E'] + SHIFTS['W']

TC_SHIFTS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

LOCS_README = """dict of whitelists for location types, including
                 single/two-qubit gates, X/Z preparations,
                 and X/Z measurements, with other special gates
                 in for good measure"""
LOCS = dict()
LOCS['SINGLE_GATES'] = ['I', 'H', 'P',
                        'X90', 'Y90', 'Z90',
                        'X', 'Y', 'Z',
                        'X180', 'Y180', 'Z180']
LOCS['DOUBLE_GATES'] = ['CNOT', 'CPHASE', 'ZZ90', 'SWAP']
LOCS['PREPARATIONS'] = ['P_X', 'P_Z']
LOCS['MEASUREMENTS'] = ['M_X', 'M_Z']

#---------------------------------------------------------------------#

class PCLayout(object):
    """
    Planar Code Layout with smooth/rough boundaries.
    """
    def __init__(self, dx, dy=None):
        dy = dy if dy else dx
        self.dx = dx
        self.dy = dy

class TCLayout(object):
    """
    Closed boundary conditions are the best.
    """
    def __init__(self, l):
        self.l = l #l-by-l tiling of the torus

        self.datas = tuple(sorted(even_odds(l, l) + odd_evens(l, l)))
        self.ancillas = {
                            'X' : tuple(sorted(even_evens(l, l))),
                            'Z' : tuple(sorted(odd_odds(l, l)))
                        }
        bits = self.datas + sum(self.ancillas.values(), ())
        self.map = crd_to_int(bits)
        self.n = 4 * l ** 2 #total, data + ancilla

    def x_ancs(self):
        return self.ancillas['X']

    def z_ancs(self):
        return self.ancillas['Z']

    def stabilisers(self):
        """
        Sometimes it's convenient to have the stabilisers of a surface
        code, especially when doing a 2d example.
        """
        stab_dict = {'X' : {}, 'Z' : {}}

        for key in stab_dict.keys():
            for anc in self.ancillas[key]:
                d_set = [self.map[ad(anc, shft, self.l)] for shft in TC_SHIFTS]
                stab = sp.X(d_set) if key == "X" else sp.Z(d_set)
                stab_dict[key][self.map[anc]] = stab

        return stab_dict

    def logicals(self):
        x_1 = [self.map[(x, 1)] for x in _evens(self.l)]
        x_2 = [self.map[(1, y)] for y in _evens(self.l)]
        z_1 = [self.map[(x, 0)] for x in _odds(self.l)]
        z_2 = [self.map[(0, y)] for y in _odds(self.l)]
        return [sp.X(x_1), sp.X(x_2), sp.Z(z_1), sp.Z(z_2)]

    def boundary_points(*args):
        return ()

    def extractor(self):
        pass #for when we do 3D

class SCLayout(object):
    """
    wraps a bunch of lists of 2d co-ordinates that I use for producing
    surface code circuits.

    Ancilla-symmetry is broken by the convention that there is a
    weight-2 XX stabiliser at the top left.
    """
    def __init__(self, dx, dy=None, h_flip=False, v_flip=False, shift=None):

        dy = dx if dy is None else dy

        self.datas = list(it.product ( range(1, 2 * dx, 2), range(1, 2 * dy, 2)))

        anc = {'x_sq': (), 'z_sq': (), 'x_top': (), 'x_bot': (), 'z_left': (), 'z_right': ()}

        anc['x_top'] = tuple([(x, 2 * dy) for x in range(2, 2 * dx, 4)])
        anc['z_left'] = tuple([(0, y) for y in range(2 * dy - 4, 0, -4)])
        if dx % 2 == dy % 2:
            anc['z_right'] = tuple([(2 * dx, y) for y in range(4, 2 * dy, 4)])
            anc['x_bot'] = tuple([(x, 0) for x in range(2 * dx - 2, 0, -4)])
        else:
            anc['z_right'] = tuple([(2 * dx, y) for y in range(2, 2 * dy, 4)])
            anc['x_bot'] = tuple([(x, 0) for x in range(2 * dx - 4, 0, -4)])
        x_sq_anc = tuple(it.product(range(4, 2 * dx, 4), range(2 * dy - 2, 0, -4)))
        x_sq_anc += tuple(it.product(range(2, 2 * dx, 4), range(2 * dy - 4, 0, -4)))
        anc['x_sq'] = x_sq_anc

        z_sq_anc = tuple(it.product(range(2, 2 * dx, 4), range(2 * dy - 2, 0, -4)))
        z_sq_anc += tuple(it.product(range(4, 2 * dx, 4), range(2 * dy - 4, 0, -4)))
        anc['z_sq'] = z_sq_anc

        self.ancillas = anc

        self.dx = dx
        self.dy = dy
        self.n = 2 * dx * dy - 1
        self.h_flip = h_flip
        self.v_flip = v_flip

        #############################################################
        boundary = {'z_top': (), 'z_bot': (), 'x_left': (), 'x_right': ()}
        boundary['z_top'] = tuple([(x, 2 * dy) for x in range(0, 2 * dx, 4)])
        boundary['x_left'] = tuple([(0, y) for y in range(2 * dy - 2, -1, -4)])
        if dx % 2 == dy % 2:
            boundary['z_bot'] = tuple([(x, 0) for x in range(2 * dx, 0, -4)])
            boundary['x_right'] = tuple([(2 * dx, y) for y in range(2, 2 * dy + 1, 4)])
        else:
            boundary['z_bot'] = tuple([(x, 0) for x in range(2 * dx - 2, -2, -4)])
            boundary['x_right'] = tuple([(2 * dx, y) for y in range(0, 2 * dy - 1, 4)])

        self.boundary = boundary

        self.xdim = 2 * self.dx + 1
        self.ydim = 2 * self.dy + 1

        coordList = list(it.product(range(self.xdim), range(self.ydim)))
        self.crd2name = { (x,y): "N" + str(x) + str(y) for (x, y) in coordList}

        self.dList = []
        self.xList = []
        self.zList = []
        self.zBndryList = []
        self.xBndryList = []

        # Flips change co-ordinates of points
        flp = lambda crd: flip(crd, self.xdim - 1, self.ydim - 1, h_flip, v_flip)
        self.datas = list(map(flp, self.datas))
        for key in self.ancillas.keys():
            self.ancillas[key] = list(map(flp, self.ancillas[key]))
        for key in self.boundary.keys():
            self.boundary[key] = list(map(flp, self.boundary[key]))


        # Flips also change the names of ancillas
        if v_flip:
            self.ancillas['x_top'], self.ancillas['x_bot'] = self.ancillas['x_bot'], self.ancillas['x_top']
            self.boundary['z_top'], self.boundary['z_bot'] = self.boundary['z_bot'], self.boundary['z_top']
        if h_flip:
            self.ancillas['z_left'], self.ancillas['z_right'] = self.ancillas['z_right'], self.ancillas['z_left']
            self.boundary['x_left'], self.boundary['x_right'] = self.boundary['x_right'], self.boundary['x_left']

        if shift:
            move = lambda tpl: (tpl[0] + shift[0], tpl[1] + shift[1])
            self.datas = list(map(move, self.datas))
            self.ancillas = {key: list(map(move, self.ancillas[key])) for key in self.ancillas.keys()}
            self.boundary = {key: list(map(move, self.boundary[key])) for key in self.boundary.keys()}

        bits = self.datas + list(it.chain.from_iterable(self.ancillas.values()))
        self.map = crd_to_int(bits)

        for x,y in self.datas:
            self.crd2name[(x,y)] = "D" + str( self.map[ (x,y) ] ).zfill(3)
            self.dList.append(self.crd2name[(x,y)])

        for x,y in self.ancillas['x_top'] + self.ancillas['x_bot'] + self.ancillas['x_sq']:
            self.crd2name[(x,y)] = "X" + str( self.map[ (x,y) ] ).zfill(3)
            self.xList.append(self.crd2name[(x,y)])
            #print(x, " ", y, " ", self.crd2name[(x,y)])

        for x,y in self.ancillas['z_left'] + self.ancillas['z_right'] + self.ancillas['z_sq']:
            self.crd2name[(x,y)] = "Z" + str( self.map[ (x,y) ] ).zfill(3)
            self.zList.append(self.crd2name[(x,y)])

        ## upper boundary points
        #for x,y in self.boundary['z_top']:
            #self.crd2name[(x,y)] = "BZ" + str( self.map[ (x+2,y-2) ] ).zfill(2)
            #self.zBndryList.append(self.crd2name[(x,y)])

        ## right boundary points
        #for x,y in self.boundary['x_right']:
            #self.crd2name[(x,y)] = "BX" + str( self.map[ (x-2,y-2) ] ).zfill(2)
            #self.xBndryList.append(self.crd2name[(x,y)])

        ## left boundary points
        #for x,y in self.boundary['x_left']:
            #self.crd2name[(x,y)] = "BX" + str( self.map[ (x+2,y+2) ] ).zfill(2)
            #self.xBndryList.append(self.crd2name[(x,y)])

        ## lower boundary points
        #for x,y in self.boundary['z_bot']:
            #self.crd2name[(x,y)] = "BZ" + str( self.map[ (x-2,y+2) ] ).zfill(2)
            #self.zBndryList.append(self.crd2name[(x,y)])

        if PY3:
            self.name2crd = {v: k for k, v in self.crd2name.items()}
        else:
            self.name2crd = {v: k for k, v in self.crd2name.iteritems()}

        g = nx.Graph()
        self.pos = self.name2crd

        # Add Nodes
        g.add_nodes_from( self.crd2name.values() )

        # Add Edges
        for x,y in self.datas:
            n1 = self.crd2name[(x,y)]
            if (x+1, y+1) in self.map:
                n2 = self.crd2name[(x+1,y+1)]
                g.add_edge( n1, n2 )
            if (x+1, y-1) in self.map:
                n2 = self.crd2name[(x+1,y-1)]
                g.add_edge( n1, n2 )
            if (x-1, y+1) in self.map:
                n2 = self.crd2name[(x-1,y+1)]
                g.add_edge( n1, n2 )
            if (x-1, y-1) in self.map:
                n2 = self.crd2name[(x-1,y-1)]
                g.add_edge( n1, n2 )

        self.graph = g

#=========================================================
    def Print(self):
        print("Printing SCLayout")

        for y in reversed( range(self.ydim) ):
            for x in (range(self.xdim)):
                print(self.crd2name[(x,y)],)
            print()
            print()
        print()

#=========================================================
    def Draw(self):
        print("Drawing SCLayout")

        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.dList , node_color='green', node_size=1000)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.xList , node_color='red', node_size=1000)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.zList , node_color='blue', node_size=1000)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.zBndryList , node_color='blue', alpha=0.15, node_size=1000)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.xBndryList , node_color='red', alpha=0.15, node_size=1000)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=10, font_color='white')

        nx.draw_networkx_edges(self.graph, self.pos, edge_color='black', arrows=False, style='dashed')

        plt.title( str(self.dx) + "X" + str(self.dy) + " lattice layout")
        plt.axis('on')
        plt.axis('equal')
        plt.grid('on')
        #plt.savefig("graph.png")
        plt.show()

#=========================================================
    def DrawSyndromes(self, xSyndNodes , zSyndNodes):
        print("Drawing SCLayout with syndromes")

        xcrdlist = [ self.map.inv[s] for s in xSyndNodes if s in self.map.inv ]
        xslist = [ self.crd2name[crd] for crd in xcrdlist ]

        zcrdlist = [ self.map.inv[s] for s in zSyndNodes if s in self.map.inv ]
        zslist = [ self.crd2name[crd] for crd in zcrdlist ]

        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.dList , node_color='green', alpha=0.85, node_size=1000)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.xList , node_color='red', alpha=0.85, node_size=1000)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.zList , node_color='blue', alpha=0.85, node_size=1000)
        #nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.zBndryList , node_color='blue', alpha=0.25, node_size=1000)
        #nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.xBndryList , node_color='red', alpha=0.25, node_size=1000)

        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=xslist, node_color='red', alpha=0.95, node_shape='D', node_size=1000)
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=zslist, node_color='blue', alpha=0.95,node_shape='D', node_size=1000)

        nx.draw_networkx_labels(self.graph, self.pos, font_size=10, font_color='white')
        nx.draw_networkx_edges(self.graph, self.pos, edge_color='black', arrows=False, style='dashed')

        plt.title( str(self.dx) + "X" + str(self.dy) + " lattice layout with X and Z syndromes")
        plt.axis('on')
        plt.axis('equal')
        plt.grid('on')
        #plt.savefig("graph.png")
        #plt.ion()
        plt.show()

#=========================================================
    # @property
    def x_ancs(self, dx=None):
        names = ['x_sq', 'x_top', 'x_bot']
        if dx == 0:
            names.remove('x_top')
        elif dx == 1:
            names.remove('x_bot')
        return reduce(add, [self.ancillas[key] for key in names])

    # @property
    def z_ancs(self, dx=None):
        names = ['z_sq', 'z_left', 'z_right']
        if dx == 0:
            names.remove('z_right')
        elif dx == 1:
            names.remove('z_left')
        return reduce(add, [self.ancillas[key] for key in names])

    def anc_type(self, anc):
        """
        Super-dangerous, I've assumed correct input.
        FIXME
        TODO
        """
        if isinstance(anc, int):
            anc = self.map.inv[anc]
        return 'X' if anc in self.x_ancs() else 'Z'

    def stabilisers(self):
        """
        Sometimes it's convenient to have the stabilisers of a surface
        code, especially when doing a 2d example.
        """
        x_stabs, z_stabs = {}, {}

        #TODO: Fix Copypasta, PEP8 me.

        for crd_tag, shft_tag in zip(['x_sq', 'x_top', 'x_bot'],
                                     ['A',    'S',     'N'    ]):
            for crd in self.ancillas[crd_tag]:
                pauli = sp.Pauli(
                    x_set=[self.map[ad(crd, dx)]
                    for dx in SHIFTS[shft_tag]])
                x_stabs[self.map[crd]] = pauli

        for crd_tag, shft_tag in zip(['z_sq', 'z_left', 'z_right'],
                                     ['A',    'E',      'W'    ]):
            for crd in self.ancillas[crd_tag]:
                pauli = sp.Pauli(
                    z_set=[self.map[ad(crd, dx)]
                    for dx in SHIFTS[shft_tag]])
                z_stabs[self.map[crd]] = pauli

        return {'X' : x_stabs, 'Z' : z_stabs}

    def logicals(self):
        x_set = [
                    self.map[_]
                    for _ in
                    filter(lambda pr: pr[0] == 1, self.datas)
                    ]
        z_set = [
                    self.map[_]
                    for _ in
                    filter(lambda pr: pr[1] == 1, self.datas)
                    ]
        return [sp.Pauli(x_set, []), sp.Pauli([], z_set)]

    def boundary_points(self, log_type):
        """
        Returns a set of fictional points that you can use to turn a
        boundary distance finding problem into a pairwise distance
        finding problem, with the typical IID XZ 2D scenario.
        logicals of the type 'log_type' have to traverse between pairs
        of output boundary points
        """
        # dx = self.dx
        # dy = self.dy
        # x_top = tuple([(x, 2 * dy) for x in range(0, 2 * dx, 4)])
        # z_right = tuple([(2 * dx, y) for y in range(2, 2 * dy + 1, 4)])
        # z_left = tuple([(0, y) for y in range(2 * dy - 2, -1, -4)])
        # x_bot = tuple([(x, 0) for x in range(2 * dx, 0, -4)])

        log_type = log_type.upper()

        if log_type == 'X':
            return self.boundary['z_top'] + self.boundary['z_bot']
            # return x_top + x_bot
        elif log_type == 'Z':
            return self.boundary['x_right'] + self.boundary['x_left']
            # return z_right + z_left
        else:
            raise ValueError("unknown logical type: {}".format(log_type))


    def extractor(self):
        """
        Returns a circuit for doing syndrome extraction, including:
        + Preparation at the right time (ancilla qubits are prepared
          immediately before their first CNOT gate)
        + Four CNOT timesteps in line with Tomita/Svore
        + Measurement at the right time (syndrome qubits are measured
          immediately after their last CNOT)
        """
        # Tomita/Svore six-step circuit
        t_0 = self.op_set_1('P_X', self.x_ancs(0))
        t_0 += self.op_set_1('P_Z', self.z_ancs(0))

        t_1 = self.x_cnot((1, 1), self.x_ancs(0))
        t_1 += self.z_cnot((1, 1), self.z_ancs(0))

        t_2 = self.x_cnot((-1, 1), self.x_ancs(0))
        t_2 += self.z_cnot((1, -1), self.z_ancs(0))
        t_2 += self.op_set_1('P_X', self.ancillas['x_top'])
        t_2 += self.op_set_1('P_Z', self.ancillas['z_right'])

        t_3 = self.x_cnot((1, -1), self.x_ancs(1))
        t_3 += self.z_cnot((-1, 1), self.z_ancs(1))
        t_3 += self.op_set_1('M_X', self.ancillas['x_bot'])
        t_3 += self.op_set_1('M_Z', self.ancillas['z_left'])

        t_4 = self.x_cnot((-1, -1), self.x_ancs(1))
        t_4 += self.z_cnot((-1, -1), self.z_ancs(1))

        t_5 = self.op_set_1('M_X', self.x_ancs(1))
        t_5 += self.op_set_1('M_Z', self.z_ancs(1))
        timesteps = [t_0, t_1, t_2, t_3, t_4, t_5]

        # pad with waits, assuming destructive measurement
        dat_locs = {self.map[q] for q in self.datas}
        for step in timesteps:
            step.extend([('I', q) for q in dat_locs - support(step)])

        return timesteps

    def extractor_h(self):
        """
        Returns a circuit for doing syndrome extraction, including:
        + 8 timesteps
        """
        
        t_0 = self.op_set_1('P', self.x_ancs(0))
        t_0 += self.op_set_1('P', self.z_ancs(0))

        t_1 = self.op_set_1('H', self.x_ancs(0))

        t_2 = self.x_cnot((1, 1), self.x_ancs(0))
        t_2 += self.z_cnot((1, 1), self.z_ancs(0))
        t_2 += self.op_set_1('P', self.ancillas['x_top'])
        t_2 += self.op_set_1('P', self.ancillas['z_right'])

        t_3 = self.x_cnot((-1, 1), self.x_ancs(0))
        t_3 += self.z_cnot((1, -1), self.z_ancs(0))
        t_3 += self.op_set_1('H', self.ancillas['x_top'])

        t_4 = self.x_cnot((1, -1), self.x_ancs(1))
        t_4 += self.z_cnot((-1, 1), self.z_ancs(1))
        t_4 += self.op_set_1('H', self.ancillas['x_bot'])

        t_5 = self.x_cnot((-1, -1), self.x_ancs(1))
        t_5 += self.z_cnot((-1, -1), self.z_ancs(1))
        t_5 += self.op_set_1('M', self.ancillas['x_bot'])
        t_5 += self.op_set_1('M', self.ancillas['z_left'])

        t_6 = self.op_set_1('H', self.x_ancs(1))

        t_7 = self.op_set_1('M', self.x_ancs(1))
        t_7 += self.op_set_1('M', self.z_ancs(1))
        timesteps = [t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7]

        # pad with waits, assuming destructive measurement
        dat_locs = {self.map[q] for q in self.datas}
        for step in timesteps:
            step.extend([('I', q) for q in dat_locs - support(step)])

        return timesteps
        
    def op_set_1(self, name, qs):
        return [(name, self.map[q]) for q in qs]

    def x_cnot(self, shft, lst):
        return [('CNOT', self.map[q], self.map[ad(q, shft)]) for q in lst]

    def z_cnot(self, shft, lst):
        return [('CNOT', self.map[ad(q, shft)], self.map[q]) for q in lst]

    def path_pauli(self, crd_0, crd_1, err_type):
        """
        Returns a minimum-length Pauli between two ancillas, given the
        type of error that joins the two.

        This function is awkward, because it works implicitly on the
        un-rotated surface code, first finding a "corner" (a place on
        the lattice for the path to turn 90 degrees), then producing
        two diagonal paths on the rotated lattice that go to and from
        this corner.
        """

        mid_v = diag_intersection(crd_0, crd_1, self.ancillas.values())

        pth_0, pth_1 = diag_pth(crd_0, mid_v), diag_pth(mid_v, crd_1)

        #path on lattice, uses idxs
        p = [self.map[crd] for crd in list(pth_0) + list(pth_1)]

        pl = sp.Pauli(p, []) if err_type == 'X' else sp.Pauli([], p)

        return pl

# -----------------------convenience functions-------------------------#
def support(timestep):
    """
    Qubits on which a list of gates act.
    """
    output = []
    for elem in timestep:
        output += elem[1:]
    return set(output)

def diag_pth(crd_0, crd_1):
    """
    Produces a path between two points which takes steps (\pm 2, \pm 2)
    between the two, starting (\pm 1, \pm 1) away from the first.
    """
    dx, dy = crd_1[0] - crd_0[0], crd_1[1] - crd_0[1]
    shft_x, shft_y = map(int, [copysign(1, dx), copysign(1, dy)])
    step_x, step_y = map(int, [copysign(2, dx), copysign(2, dy)])
    return zip(range(crd_0[0] + shft_x, crd_1[0], step_x),
                range(crd_0[1] + shft_y, crd_1[1], step_y))

def diag_intersection(crd_0, crd_1, ancs=None):
    """
    Uses a little linear algebra to determine where diagonal paths that
    step outward from ancilla qubits intersect.
    This allows us to reduce the problems of length-finding and
    path-making to a pair of 1D problems.
    """
    a, b, c, d = crd_0[0], crd_0[1], crd_1[0], crd_1[1]
    vs = [
            ( int(( d + c - b + a ) / 2), int(( d + c + b - a ) / 2) ),
            ( int(( d - c - b - a ) / -2), int(( -d + c - b - a ) / -2) )
        ]

    if ancs:
        if vs[0] in sum(ancs, ()):
            mid_v = vs[0]
        else:
            mid_v = vs[1]
    else:
        mid_v = vs[0]

    return mid_v

def flip(crd, dx, dy, h, v):
    return (dx - crd[0] if h else crd[0], dy - crd[1] if v else crd[1])

#---------------------------------------------------------------------#


#------------------toric code convenience functions-------------------#

grid = lambda lst_1, lst_2: map(tuple, list(it.product(lst_1, lst_2)))

_evens = lambda n: range(0, 2 * n, 2)

_odds = lambda n: range(1, 2 * n + 1, 2)

even_odds = lambda nx, ny: grid(_evens(nx), _odds(ny))

odd_evens = lambda nx, ny: grid(_odds(nx), _evens(ny))

even_evens = lambda nx, ny: grid(_evens(nx), _evens(ny))

odd_odds = lambda nx, ny: grid(_odds(nx), _odds(ny))

crd_to_int = lambda lst: bd.bidict(zip(sorted(lst), range(len(lst))))

def sym_coords(nx, ny):
    """
    Convenience function for square lattice definition, returns all
    pairs of co-ordinates on an n-by-n lattice which are both even or
    both odd. Note that it iterates over all of the points in the 2D
    grid which is nx-by-ny large, this is so the list of returned
    coordinates is sorted.
    """
    symmetric_coordinates = []
    for x in range(2 * nx):
        if x % 2 == 0:
            for y in range(2 * ny):
                if y % 2 == 0:
                    symmetric_coordinates.append((x, y))
        else:
            for y in range(2 * ny):
                if y % 2 == 1:
                    symmetric_coordinates.append((x, y))
    return symmetric_coordinates


def skew_coords(nx, ny):
    """
    Convenience function for square lattice definition, returns all
    pairs of co-ordinates on an n-by-n lattice which are "even-odd" or
    "odd-even".Note that it iterates over all of the points in the 2D
    grid which is nx-by-ny large, this is so the list of returned
    coordinates is sorted.
    """
    skewed_coordinates = []
    for x in range(2 * nx):
        if x % 2 == 0:
            for y in range(2 * ny):
                if y % 2 == 1:
                    skewed_coordinates.append((x, y))
        else:
            for y in range(2 * ny):
                if y % 2 == 0:
                    skewed_coordinates.append((x, y))
    return skewed_coordinates


def all_coords(nx, ny):
    """
    Returns all points on a square lattice, used to determine the dual
    lattice of the SquareOctagonLattice.
    """
    all_coordinates = []
    for x in range(2 * nx):
        for y in range(2 * ny):
            all_coordinates.append((x, y))
    return all_coordinates

#---------------------------------------------------------------------#
