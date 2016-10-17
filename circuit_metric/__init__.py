from sys import version_info
if version_info[0] == 3:
    from .circuit_metric import *
elif version_info[0] == 2:
    from circuit_metric import *
