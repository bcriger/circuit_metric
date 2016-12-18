import bidict as bd, itertools as it
from layout_utils import *
from math import copysign

README = """
There are a lot of options for a clustered layout for the surface code.
It could get very messy if I'm not careful. 
I don't want to end up with a bunch of options in a class construction,
and I also don't want to end up with repetitive class declarations.

The overall goal of each of these layouts is to keep long-range (slow, 
low-fidelity) gates off the data qubits, allowing me to use single dots
for the data qubits (is this better?) and compensate for noise in that 
gate using majority voting or signal processing techniques on the
ancillas it affects.
"""

HELPER_SHFTS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

class FiveFourLayout(object):
    """
    Clustered layout for the surface code which uses two to five qubits
    per cluster.
    Data clusters in the bulk have five qubits; one for data and one to
    connect to the ancilla for each neighbouring stabiliser (2 X, 2 Z).
    Data clusters on a boundary are missing a qubit, since they only
    have three neighbouring stabilisers, likewise data clusters at a 
    corner have two ancilla qubits.
    Ancilla clusters in the bulk have four qubits, one per neighbour. 
    Ancilla clusters at the boundary have two neighbours, so two
    qubits.  
    """
    def __init__(self, dx, dy=None):
        
        dy = dy if dy else dx
        
        self.dx = dx
        self.dy = dy
        
        # Derived Properties -- Qubit Locations
        self.datas = list(it.product(range(2, 5 * dx - 2, 5),
                                        range(2, 5 * dy - 2, 5)))
        data_helpers = []
        for d in self.datas:
            data_helpers.extend([ad(d, s) for s in HELPER_SHFTS])    
        
        #remove unwanted bits (ugly, inefficient)
        x_max, y_max = 5 * dx - 2, 5 * dy - 2
        
        n_bnd = [(x, y_max) for x in
                    range(1, x_max - 1, 10) + range(8, x_max - 1, 10)]
        
        w_bnd = [(1, y) for y in
                    range(1, y_max - 1, 10) + range(8, y_max - 1, 10)]
        
        e_bnd = [(x_max, y) for y in
                    range(3, y_max + 1, 10) + range(6, y_max + 1, 10)]
        
        s_bnd = [(x, 1) for x in
                    range(3, x_max + 1, 10) + range(6, x_max + 1, 10)]

        rejects = n_bnd + s_bnd + e_bnd + w_bnd

        data_helpers = filter(lambda h: h not in rejects, data_helpers)
        self.data_helpers = data_helpers

        x_ancs = []
        
#---------------------------------------------------------------------#
