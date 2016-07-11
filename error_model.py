import numpy as np
import sparse_pauli as sp

TOL = 10.**-14

class PauliErrorModel(object):
    """
    Small, dense Pauli Error Model. 

    Wraps an array and a set.

    The array must be flat with 4**n real entries for n the length of
    the set `qs`. These entries must sum to 1 to within tolerance TOL
    (set to 10**-14 by default).
    
    All the set has to do is cast to a set.

    """
    def __init__(self, p_arr, qs):
        
        try: 
            p_arr = np.array(p_arr)
        except Exception as err:
            raise TypeError("Input array of probabilities ({}) "
                "does not cast to array.  ".format(p_arr) + 
                "Error: " + err.strerror)
        
        if type(qs) is set:
            raise TypeError("THOU SHALT NOT USE UN-ORDERED TYPES FOR "
                "QUBIT LISTS")

        try: 
            qs = list(qs)
        except Exception as err:
            raise TypeError("Input set of qubits ({}) "
                "does not cast to list.  ".format(qs) + 
                "Error: " + err.strerror)
        
        delta = abs(sum(p_arr) - 1.)
        if delta > TOL:
            raise ValueError("Input probabilities must sum to" + 
                " unity. \n    Input: {}".format(p_arr) + 
                "\n    Difference: {}".format(delta))
        
        if len(p_arr) != 4 ** len(qs):
            raise ValueError("Number of qubits inconsistent. Qubit "
                "labels imply {} qubit(s), but".format(len(qs)) + 
                " there are {} probabilities.".format(len(p_arr)))

        self.p_arr = p_arr
        self.qs = qs

    def sample(self):
        """
        Produces a random number between 0 and 1, and iteratively 
        subtracts until the difference is less than the probability of
        a Pauli in the model, and returns that Pauli.
        I use int_sample to get an index from the array p_arr, then
        convert to a sparse_pauli using int_to_pauli. 
        """
        indx = int_sample(self.p_arr)
        return int_to_pauli(indx, self.qs)

def int_sample(probs):
    
    value = np.random.rand()
    
    for idx, p in enumerate(probs):
        if value < p:
            return idx
        else:
            value -= p
    
    raise ValueError("Enumeration Failure (probability > 1?)") 

def int_to_pauli(intgr, lbl_lst):
    """
    This one needs a little explaining. I convert every integer with
    an even number of bits to a Pauli by splitting the integer into a
    left (big) half and a right (small) half. 
    Locations where the left half is 1 are elements of the `x_set`.
    Locations where the right half is 1 are elements of the `z_set`.
    """
    bits = bin(intgr)[2:].zfill(2 * len(lbl_lst))
    xs, zs = bits[ : len(lbl_lst)], bits[len(lbl_lst) : ]
    return sp.Pauli({l for l, b in zip(lbl_lst, xs) if b == '1'},
                    {l for l, b in zip(lbl_lst, zs) if b == '1'})