import bidict as bd
import itertools as it


class TestClass:
    """
    Verification of syndrome change based on specific errors
    """
    def __init__(self):
        self.dataq = [(1, 1)]
        self.ancq = {'x_anc': ((2, 2),)}
        testbits = self.dataq + list(it.chain.from_iterable(self.ancq.values()))
        self.map = bd.bidict(zip(sorted(testbits), range(len(testbits))))

    def x_ancs(self):
        names = ['x_anc', ]
        return [(self.ancq['x_anc'],),]

    def extractor(self):
        """
        circuit that describes the test :
        + Preparation of ancilla qubit
        + CNOT between the data qubit (control) and the ancilla qubit (target)
        + Measurement of the ancilla qubit in the Z basis and print the outcome
        + Repeat the 3 operations one more time

        :return: timesteps
        """
        t_0 = self.op_set_1('P_Z', self.ancq['x_anc'])
        t_1 = [('CNOT', 0, 1), ]
        t_2 = self.op_set_1('M_Z', self.ancq['x_anc'])
        timesteps = [t_0, t_1, t_2, t_0, t_1, t_2]

        # pad with waits, assuming destructive measurement
        dat_locs = {self.map[q] for q in self.dataq}
        for step in timesteps:
            step.extend([('I', q) for q in dat_locs - support(step)])

        return timesteps

    def op_set_1(self, name, qs):
        return [(name, self.map[q]) for q in qs]

    def x_cnot(self, shft, lst):    #(CNOT, 0, 1)
        return [('CNOT', self.map[q], self.map[ad(q, shft)]) for q in lst]

# -----------------------convenience functions-------------------------#
ad = lambda tpl_0, tpl_1: tuple(a + b for a, b, in zip(tpl_0, tpl_1))


def support(timestep):
    output = []
    for elem in timestep:
        output += elem[1:]
    return set(output)

