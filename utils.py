README = """
This file contains miscellaneous code which is useful for checking 
model_to_metric against hand calculations, but shouldn't necesarily
be part of the codebase that we use to find edge weights.
"""

from model_to_metric import *

DIST_2_LABELS = {
                    ((2, 2, 0),)                                : ((6, 7),),
                    ((2, 4, 0),)                                : ((1, 2),),
                    ((2, 0, 0), (2, 2, 0))                      : ((0, 2), (6, 7)),
                    ((2, 0, 0),)                                : ((0, 2),),
                    ((2, 4, 0), (2, 2, 0))                      : ((1, 2), (6, 7)),
                    ((2, 0, 0), (2, 0, 1))                      : ((0, 3),),
                    ((2, 2, 0), (2, 2, 1))                      : ((6, 8),),
                    ((2, 0, 0), (2, 2, 0), (2, 0, 1))           : ((0, 3), (6, 7)),
                    ((2, 2, 0), (2, 0, 1))                      : ((3, 5), (6, 7)),
                    ((2, 0, 1),)                                : ((3, 5),),
                    ((2, 2, 1),)                                : ((8, 9),),
                    ((2, 4, 0), (2, 2, 0), (2, 2, 1))           : ((1, 2), (6, 8)),
                    ((2, 4, 0), (2, 2, 1))                      : ((1, 2), (8, 9)),
                    ()                                          : (),
                    ((2, 4, 0), (2, 2, 0), (2, 2, 1), (2, 0, 1)): ((1, 3), (6, 8)),
                    ((2, 4, 0), (2, 0, 1))                      : ((1, 3),),
                    ((2, 4, 0), (2, 4, 1))                      : ((1, 4),),
                    ((2, 4, 0), (2, 0, 1), (2, 2, 1))           : ((1, 3), (8, 9)),
                    ((2, 4, 0), (2, 2, 0), (2, 0, 1))           : ((1, 3), (6, 7)),
                    ((2, 2, 0), (2, 2, 1), (2, 0, 1))           : ((6, 8), (3, 5)),
                    ((2, 0, 1), (2, 2, 1))                      : ((3, 5), (8, 9)),
                    ((2, 4, 1),)                                : ((4, 5),),
                    ((2, 4, 0), (2, 4, 1), (2, 2, 1))           : ((1, 4), (8, 9)),
                    ((2, 4, 1), (2, 2, 1))                      : ((4, 5), (8, 9))
                }

DIST_2_LABELS_DOC = """
Map from syndrome pairs to edge labels from Savvas' hand calculation.
This is a large, handwritten object, and is therefore likely to contain errors.
"""

def errors_and_syndromes(f_ps, circ, layout):
    """
    Returns a time-ordered list of errors that can occur 
    after a given timestep (or in the case of measurement errors,
    *before the following timestep*), given the error model, extractor,
    and layout.
    """
    lst = []
    for t in range(len(f_ps)):
        step_dict = {
                        f: synds_to_changes(layout, synd_set(circ, f, t))
                        for f, p in f_ps[t]
                    }

        lst.append(step_dict)

    return lst

def edge_count(err_edge, edge_lbl):
    """
    Put in the output from errors_and_syndromes, along with a 
    descriptor for the edge (a subset of the activated syndromes in my 
    notation, or a pair of vertices in Savvas' notation) and get out 
    the number of times that descriptor shows up.
    """
    return sum([
                quantify(dct.values(), lambda elem: edge_lbl in elem)
                for dct in err_edge
                ])

