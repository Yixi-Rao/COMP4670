import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cho_solve, cholesky, solve_triangular

l  = 2.5858691508062996 

v  = 0.7529170173330241

X  = np.array([[-0.5],
               [2.2]])
Y  = np.array([[-1.31584035],
               [-0.09311054]])

D_XY  = cdist(X, X, 'euclidean')
my_Ky = v * (1 + ((np.sqrt(5) / l) + (5 / (3 * l ** 2)) * D_XY) * D_XY) * (np.exp(-(np.sqrt(5) / l) * D_XY))
# Ky = np.array([[0.75291702, 0.37560582],
#                [0.37560582, 0.75291702]])
L     = cholesky(my_Ky, lower=True)
alpha = cho_solve((L.T, L), Y) 
n     = Y.shape[0]
r = ((-1 / 2) * Y.T @ alpha)[0][0] - np.sum([np.log(L[i, i]) for i in range(L.shape[0])]) - (n / 2) * np.log(2 * np.pi)

print(D_XY)
print(my_Ky)
print(L)
print(alpha)
print(r)