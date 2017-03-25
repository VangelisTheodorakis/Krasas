import numpy as np
from numpy.linalg import inv
from numpy import transpose as tr
"""
X array is D x N

"""


def do_PCA(x_array):
    hat_x_array = np.empty([x_array.shape[0], x_array.shape[1]])

    for data_column in range(0, x_array.shape[1]):
        print("test %s", np.mean(x_array[:,data_column]))
        hat_x_array[:, data_column] = np.mean(x_array[:,data_column])