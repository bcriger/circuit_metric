import itertools as it
from operator import add
import bidict as bd

class SCLayout(object):
    """
    wraps a bunch of lists of 2d co-ordinates that I use for producing
    surface code circuits.
    """
    def __init__(self, d):
        
        self.datas = list(it.product(range(1, 2 * d, 2), repeat=2))
        
        anc = {'x_sq': (), 'z_sq': (), 'x_top': (),
                'x_bot': (), 'z_left': (), 'z_right': ()}
        
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


    def extractor(self):
        """
        Returns a circuit for doing syndrome extraction, including:
        + Preparation at the right time (ancilla qubits are prepared
          immediately before their first CNOT gate)
        + Four CNOT timesteps in line with Tomita/Svore
        + Measurement at the right time (syndrome qubits are measured 
          immediately after their last CNOT)
        """       
        #Tomita/Svore six-step circuit
        t_0  = self.op_set_1('P_X', self.x_ancs(0))
        t_0 += self.op_set_1('P_Z', self.z_ancs(0))

        t_1  = self.x_cnot((1, 1), self.x_ancs(0))
        t_1 += self.z_cnot((1, 1), self.z_ancs(0))
        
        t_2  = self.x_cnot((-1, 1), self.x_ancs(0))
        t_2 += self.z_cnot((1, -1), self.z_ancs(0))
        t_2 += self.op_set_1('P_X', self.ancillas['x_top'])
        t_2 += self.op_set_1('P_Z', self.ancillas['z_right'])
        
        t_3  = self.x_cnot((1, -1), self.x_ancs(1))
        t_3 += self.z_cnot((-1, 1), self.z_ancs(1))
        t_3 += self.op_set_1('M_X', self.ancillas['x_bot'])
        t_3 += self.op_set_1('M_Z', self.ancillas['z_left'])
        
        t_4  = self.x_cnot((-1, -1), self.x_ancs(1))
        t_4 += self.z_cnot((-1, -1), self.z_ancs(1))
        
        t_5  = self.op_set_1('M_X', self.x_ancs(1))
        t_5 += self.op_set_1('M_Z', self.z_ancs(1))
        timesteps = [t_0, t_1, t_2, t_3, t_4, t_5]
        
        #pad with waits, assuming destructive measurement
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

#-----------------------convenience functions-------------------------#
ad = lambda tpl_0, tpl_1: tuple(a + b for a, b, in zip(tpl_0, tpl_1))

def support(timestep):
    output = []
    for elem in timestep:
        output += elem[1:]
    return set(output)

#---------------------------------------------------------------------#
