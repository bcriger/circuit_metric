import model_to_metric as md

f_ps, circ, layout = md.fault_probs1()
print f_ps
#layout = sc.SCLayout(3)
#circ = layout.extractor()
# output = md.model_to_pairs(f_ps, circ, layout)
#print 'output : ', output
# X_synd = md.css_pairs(output, layout, 'X')
#print 'X_synd = ', X_synd
# Z_synd = md.css_pairs(output, layout, 'Z')
#print 'Z_synd = ', Z_synd
# verticesX, edgesX, weightsX = md.dict_to_metric(X_synd)
# print 'VerticesX = ', verticesX
# print 'EdgesX = ', edgesX
# print 'WeightsX = ', weightsX
# verticesZ, edgesZ, weightsZ = md.dict_to_metric(Z_synd)
# print 'VerticesZ = ', verticesZ
# print 'EdgesZ = ', edgesZ
# print 'WeightsZ = ', weightsZ


