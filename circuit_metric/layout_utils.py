def ad(t_0, t_1, l=None):
    return tuple((a + b) % (2 * l) if l else (a + b) for a, b, in zip(t_0, t_1))
