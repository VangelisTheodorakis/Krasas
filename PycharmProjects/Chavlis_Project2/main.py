from kmeans import *
from numpy import transpose as tr
print("204 Methods")


m1, cov1 = [1, 1], [[0.5, 0], [0, 0.5]]
m2, cov2 = [-1, -1], [[0.75, 0], [0, 0.75]]

data = np.random.multivariate_normal(m1, cov1, 220)
data2 = np.random.multivariate_normal(m2, cov2, 280)
print(data)

np.append(data, data2)

centroids , C = my_kmeans(data, 2, 'distance', 100, 1)

print("Data shape ", data.shape)

show(data, C, centroids)
