import itertools as it
from operator import add
import bidict as bd
import sparse_pauli as sp 


#------------------------------constants------------------------------#
SHIFTS = {
            'N': ((-1, 1), (1, 1)),
            'E': ((1, 1), (1, -1)),
            'W': ((-1, 1), (-1, -1)),
            'S': ((1, -1), (-1, -1))
            }
SHIFTS['A'] = SHIFTS['E'] + SHIFTS['W']
SHIFTS_README = """(dx, dy) so that, given an ancilla co-ordinate
                   (x, y), there will be data qubits at
                   (x + dx, y + dy)."""
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
LOCS['PREPARATIONS'] = ['P_X, P_Z']
LOCS['MEASUREMENTS'] = ['M_X, M_Z']
#---------------------------------------------------------------------#


class SCLayout(object):
    """
    wraps a bunch of lists of 2d co-ordinates that I use for producing
    surface code circuits.
    """
    def __init__(self, d):
        
        self.datas = list(it.product(range(1, 2 * d, 2), repeat=2))

        anc = {'x_sq': (), 'z_sq': (), 'x_top': (), 'x_bot': (), 'z_left': (), 'z_right': ()}

        anc['x_top'] = tuple([(x, 2 * d) for x in range(2, 2 * d, 4)])
        anc['z_right'] = tuple([(2 * d, y) for y in range(4, 2 * d, 4)])
        anc['z_left'] = tuple([(0, y) for y in range(2 * d - 4, 0, -4)])
        anc['x_bot'] = tuple([(x, 0) for x in range(2 * d - 2, 0, -4)])
        x_sq_anc = tuple(it.product(range(4, 2 * d, 4),
                                    range(2 * d - 2, 0, -4)))
        x_sq_anc += tuple(it.product(range(2, 2 * d, 4), 
                                     range(2 * d - 4, 0, -4)))
        anc['x_sq'] = x_sq_anc

        z_sq_anc = tuple(it.product(range(2, 2 * d, 4), 
                                    range(2 * d - 2, 0, -4)))
        z_sq_anc += tuple(it.product(range(4, 2 * d, 4), 
                                     range(2 * d - 4, 0, -4)))
        anc['z_sq'] = z_sq_anc
        self.ancillas = anc
        
        bits = self.datas + list(it.chain.from_iterable(anc.values()))

        self.map = bd.bidict(zip(sorted(bits), range(len(bits))))
        self.d = d

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

    def boundary_points(self, anc_type):
        """
        Returns a set of fictional points that you can use to turn a 
        boundary distance finding problem into a pairwise distance 
        finding problem, with the typical IID XZ 2D scenario.

        This function is rendered more-or-less obsolete by the 'missing
        tiles' method that comes from Bombin 2007. I don't call it 
        anywhere, and neither should you.

        TODO FIXME
        """
        d = self.d
        z_top = tuple([(x, 2 * d) for x in range(4, 2 * d, 4)])
        x_right = tuple([(2 * d, y) for y in range(2, 2 * d, 4)])
        x_left = tuple([(0, y) for y in range(2 * d - 2, 0, -4)])
        z_bot = tuple([(x, 0) for x in range(2 * d - 4, 0, -4)])
        return z_top + z_bot if anc_type == 'Z' else x_right + x_left

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

    def op_set_1(self, name, qs):
        return [(name, self.map[q]) for q in qs]

    def x_cnot(self, shft, lst): 
        return [('CNOT', self.map[q], self.map[ad(q, shft)]) for q in lst]

    def z_cnot(self, shft, lst): 
        return [('CNOT', self.map[ad(q, shft)], self.map[q]) for q in lst]

# -----------------------convenience functions-------------------------#
ad = lambda tpl_0, tpl_1: tuple(a + b for a, b, in zip(tpl_0, tpl_1))


def support(timestep):
    output = []
    for elem in timestep:
        output += elem[1:]
    return set(output)

# ---------------------------------------------------------------------#

