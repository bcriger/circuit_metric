import model_to_metric as md
import SCLayoutClass as sc
'''
s = sc.SCLayout(3)
print('data qubits : ', s.datas)
print('ancilla qubits : ', s.ancillas)
print('X ancilla qubits : ', s.x_ancs())
print('Z ancilla qubits : ', s.z_ancs())
print('extractor : ', s.extractor())
print('map : ', s.map)
'''


'''
dicto = {'a': 1, 'b': 2}
print(dicto.keys())
print(dicto.items())
print(dicto.values())
dicto.update({'c': 3})
print(dicto.items())
'''
'''
list_prob = [0.61, 0.53, 0.64, 0.584, 0.59, 0.6]
list_yes = []  # zero yesses, so everything is taken as 1-prob
res = md.set_prob(list_prob, list_yes)
print('probability of an event = ', res)

list_yes = [0]  # the first prob has a yes, so prob*(1-prob)*...*(1-prob)
res = md.set_prob(list_prob, list_yes)
print('probability of an event = ', res)

list_yes = [0, 4]
res = md.set_prob(list_prob, list_yes)
print('probability of an event = ', res)
'''
'''
list_prob = [0.61, 0.53, 0.64, 0.584, 0.59, 0.6]
res = md.r_event_prob(list_prob, 0)  # (1-p1 * 1-p2 * 1-p3 * 1-p4 * 1-p5 * 1-p6)
print(res)
res = md.r_event_prob(list_prob, 1) # (p1 * 1-p2 * 1-p3 * 1-p4 * 1-p5 * 1-p6) + (1-p1 * p2 * 1-p3 * 1-p4 * 1-p5 * 1-p6)+
                                    # (1-p1 * 1-p2 * p3 * 1-p4 * 1-p5 * 1-p6) + (1-p1 * 1-p2 * 1-p3 * p4 * 1-p5 * 1-p6)+
                                    # (1-p1 * 1-p2 * 1-p3 * 1-p4 * p5 * 1-p6) + (1-p1 * 1-p2 * 1-p3 * 1-p4 * 1-p5 * p6)
print(res)
res = md.r_event_prob(list_prob, 2)
print(res)
res = md.r_event_prob(list_prob, 3)
print(res)
res = md.r_event_prob(list_prob, 4)
print(res)
res = md.r_event_prob(list_prob, 5)  # (1-p1 * p2 * p3 * p4 * p5 * p6) + (p1 * 1-p2 * p3 * p4 * p5 * p6)+
                                     # (p1 * p2 * 1-p3 * p4 * p5 * p6) + (p1 * p2 * p3 * 1-p4 * p5 * p6)+
                                     # (p1 * p2 * p3 * p4 * 1-p5 * p6) + (p1 * p2 * p3 * p4 * p5 * 1-p6)

print(res)
res = md.r_event_prob(list_prob, 6)  # p1 * p2 * p3 * p4 * p5 * p6
print(res)
'''

'''
pair_p_dict = {(1, 2): [0.5, 0.3, 0.2, 0.1], (1, 3): [0.4, 0.1, 0.9]}
[v, e, w] = md.dict_to_metric(pair_p_dict)
print(v)
print(e)
print(w)
'''


#sc = sc.SCLayout(3)     # creation of distance 3 code layout
#circ = sc.extractor()   # creation of the circuit that corresponds to the distance 3 code
# print('circ : ', circ)
# t = md.loc_type(circ[1], 'I')   # filters out the timesteps that include the provided string
# print(t)

# print('-------')
# print(md.prep_faults(circ))     # inserts errors on all the ancilla qubits that are being prepared
#
# print('-------')
# print(md.meas_faults(circ))     # inserts errors on all the ancilla qubits that are being measured
#
# print('+++++++++++++')
# print(md.prop_circ(circ))       #
#
# print('+++++++++++++')
# md.prop_circ(circ, True)

# print('+++++++++++++')
# print(md.str_faults(circ, 'P_'))

'''
pair_p_dict = {(0, 5): [0.01], (0, 11): [0.1], (5, 11): [0.01], (5, 16): [0.1], (11, 16): [0.01],
               (6, 12): [0.1], (6, 4): [0.01], (12, 4): [0.01], (12, 10): [0.1], (4, 10): [0.01]}
[v, e, w] = md.dict_to_metric(pair_p_dict)
print(v)
print(e)
print(w)

syndromes = md.synd_set(circ, [], 1)
print 'syndromes = ', syndromes
'''
f_ps = md.fault_probs(3, False)
layout = sc.SCLayout(3)
circ = layout.extractor()
output = md.model_to_pairs(f_ps, circ, layout)
# print 'output : ', output

# [
#     [('P_X', 12), ('P_X', 4), ('P_X', 10), ('P_Z', 5), ('P_Z', 11), ('P_Z', 0), ('I', 1), ('I', 2), ('I', 3), ('I', 7), ('I', 8), ('I', 9), ('I', 13), ('I', 14), ('I', 15)],
#     [('CNOT', 12, 15), ('CNOT', 4, 8), ('CNOT', 10, 13), ('CNOT', 9, 5), ('CNOT', 14, 11), ('CNOT', 2, 0), ('I', 1), ('I', 3), ('I', 7)],
#     [('CNOT', 12, 9), ('CNOT', 4, 2), ('CNOT', 10, 7), ('CNOT', 8, 5), ('CNOT', 13, 11), ('CNOT', 1, 0), ('P_X', 6), ('P_Z', 16), ('I', 3), ('I', 14), ('I', 15)],
#     [('CNOT', 12, 14), ('CNOT', 4, 7), ('CNOT', 6, 9), ('CNOT', 3, 5), ('CNOT', 8, 11), ('CNOT', 15, 16), ('M_X', 10), ('M_Z', 0), ('I', 1), ('I', 2), ('I', 13)],
#     [('CNOT', 12, 8), ('CNOT', 4, 1), ('CNOT', 6, 3), ('CNOT', 2, 5), ('CNOT', 7, 11), ('CNOT', 14, 16), ('I', 9), ('I', 13), ('I', 15)],
#     [('M_X', 12), ('M_X', 4), ('M_X', 6), ('M_Z', 5), ('M_Z', 11), ('M_Z', 16), ('I', 1), ('I', 2), ('I', 3), ('I', 7), ('I', 8), ('I', 9), ('I', 13), ('I', 14), ('I', 15)]
# ]
