import numpy as np
from numpy.linalg import inv
from numpy import transpose as tr
"""
W matrix D x M  where M is the size of the final wanted dimension
M matrix M x M
X matrix D x N

"""


class PPCA:
    def __init__(self, final_dimensions=2, sigma=1.0):
        self.w_old = 0
        self.w_new = 0
        self.final_dimensions = final_dimensions
        self.sigma_old = 0.0
        self.sigma_new = 0.0
        self.hat_x = 0
        self.matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        self.rows = 0
        self.columns = 0
        self.sigma = sigma

    def fit(self):
        print("--->", self.matrix.shape[0])
        print("--->", self.matrix.shape[1])

        # D
        self.rows = self.matrix.shape[0]
        # N

        self.columns = self.matrix.shape[1]
        self.__fit_em()

    def __fit_em(self, maxit=20):

        # W is a DxM array
        w_old = np.random.rand(self.rows, self.final_dimensions)

        # Initialize array Dx1
        hat_x_array = np.empty([self.rows, 1])

        # Construct the array with the mean of each row
        for data_line in range(0, self.rows):
            print("test %s", np.mean(self.matrix[data_line]))
            hat_x_array[data_line, :] = np.mean(self.matrix[data_line])

        print("Hat x array shape %s", hat_x_array.shape)
        print("Hat x array %s", hat_x_array)
        # Copy data array
        normalized_x = self.matrix

        print("Normalized_x array size %s", normalized_x.shape)
        print("Normalized_x array %s", normalized_x)

        hat_x_array = np.repeat(hat_x_array, 4, axis=1)

        print("Hat x array %s", hat_x_array)

        normalized_x = np.subtract(normalized_x.T, hat_x_array.T)

        normalized_x = normalized_x.T
        print("Normalized_x array after subtraction %s, shape is %s", normalized_x, normalized_x.shape)

        # The old sigma
        sigma_old = self.sigma_old

        # The sigma square
        sigma_square = self.sigma**2

        M = tr(w_old).dot(w_old)+sigma_square*np.eye(self.final_dimensions)

        for iteration in range(maxit):
            print("M %s", inv(M).shape)

            print("W old %s",w_old.shape)
            print("Normalized %s", normalized_x.shape)
            print("++++++++")

            E_z = inv(M).dot(tr(w_old)).dot(normalized_x)
            E_z_z_transpose = sigma_square * inv(M) + E_z.dot(tr(E_z))

            w_new_left = 0
            w_new_right = 0
            sigma_square_new = 0

            print("E z%s", E_z.shape)
            print("E z z transpose  %s", E_z_z_transpose.shape)

            print("++++++++")

            for n in range(0, self.final_dimensions):
                print("Normalized x is %s and shape is %s", normalized_x[:,n], normalized_x[:,n].reshape(self.rows, 1).shape)
                w_new_left += normalized_x[:,n].reshape(self.rows, 1).dot(tr(E_z)[n].reshape(1, self.final_dimensions))
                print("w_new_left---> %s",w_new_left.shape)
                w_new_right += E_z_z_transpose[:,n].reshape(1, self.final_dimensions)**-1
                print("w_new_right---> %s", w_new_right.shape)

            w_new = w_new_left * w_new_right

            for n in range(0, self.final_dimensions):
                sigma_square_new += np.linalg.norm(normalized_x[:, n].reshape(self.rows, 1)) ** 2 \
                                    - 2 * tr(E_z[:, n]).reshape(self.final_dimensions, 1)\
                                    .dot(w_new[:, n].reshape(1, self.rows))
                                    #.dot(normalized_x[:,n].reshape(self.rows, 1))

            sigma_square = sigma_square_new

            w_old = np.array(w_new)
            print("w new --> %s",w_new.shape)
        return
