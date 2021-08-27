import numpy as np
import scipy.sparse
import sympy
import sklearn.datasets
import sklearn.feature_extraction.text
import umap
import umap.plot
import matplotlib.pyplot as plt
from numpy import genfromtxt

primes = list(sympy.primerange(2, 11000))
prime_to_column = {p:i for i, p in enumerate(primes)}

ssim = genfromtxt('../Temp/SSIM_all/ssmi-all.csv', delimiter=',')
n1 = 64
n2 = n1**2

lil_matrix_rows = []
lil_matrix_data = []

for n in range(10000):
    prime_factors = sympy.primefactors(n)
    lil_matrix_rows.append([prime_to_column[p] for p in prime_factors])
    lil_matrix_data.append([1] * len(prime_factors))



factor_matrix = scipy.sparse.lil_matrix((len(lil_matrix_rows), len(primes)), dtype=np.float32)
factor_matrix.rows = np.array(lil_matrix_rows)
factor_matrix.data = np.array(lil_matrix_data)
print(factor_matrix.shape)


mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True).fit(factor_matrix)

umap.plot.points(mapper, values=np.arange(10000), theme='viridis')
umap.plot.plt.show()
