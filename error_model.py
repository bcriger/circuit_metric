# import sparse_pauli as p
import qecc as q
import itertools as it

# ------------------------Internal Functions---------------------------#


def powerset(iterable):
    """
    Returns sorted iterables onto all subsets of elements in an
    iterable.
    """
    s = list(iterable)
    rs = range(len(s)+1)
    return it.chain.from_iterable(it.combinations(s, r) for r in rs)


def all_paulis(qs=None, nq=None):
    """
    This function returns a list of all possible Paulis on a set of
    qubits. There are two ways to enter the set; either using a set of
    keys `qs` or an integer `nq` (the set of qubits is then taken to be
    `range(nq)`).
    """
    if (qs is None) == (nq is None):
        raise ValueError("Exactly one of `qs` or `nq` must be set. "
                         "You set qs : {}, nq : {}".format(qs, nq))
    qs = range(nq) if qs is None else qs
    # this is very memory intensive, but I'm counting on sets of length
    # one or two being input.
    sbsts = list(powerset(qs))

    return [q.Pauli.from_sparse.Pauli(x, z) for x, z in it.product(sbsts, sbsts)]
    # return [p.Pauli(x, z) for x, z in it.product(sbstsall , sbsts)]
# ---------------------------------------------------------------------#

# -----------------------------Main Class------------------------------#

# ---------------------------------------------------------------------#
