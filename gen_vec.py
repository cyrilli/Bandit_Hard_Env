import numpy as np
from util_functions import featureUniform, gaussianFeature

dimension = 25
argv = {'l2_limit': 1}
v = gaussianFeature(dimension, argv=argv)
l2_norm = np.linalg.norm(v, ord=2)
v = v/l2_norm

e1 = np.eye(dimension)[0]
if v[0] >= 0:
    w = v+e1
else:
    w = v-e1
Q = np.identity(dimension) - 2 * np.outer(w, w)/np.dot(w, w)
assert Q.shape == (dimension, dimension)

for i in Q[1:]:
    print(np.dot(i, v))
    print(np.linalg.norm(i, ord=2))