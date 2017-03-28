import numpy as np
from numpy.linalg import inv
from numpy import transpose as tr
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, factor=.3, noise=.05)

class PPCA:
    def __init__(self, final_dimensions=2, sigma=0):
        self.w_old = 0
        self.w_new = 0
        self.final_dimensions = final_dimensions
        self.sigma_old = 0.0
        self.sigma_new = 0.0
        self.hat_x = 0
        self.matrix = X
        self.rows = 0
        self.columns = 0
        self.sigma = sigma

    def fit(self):
        print("--->", self.matrix.shape[0])
        print("--->", self.matrix.shape[1])

        # N
        self.rows = self.matrix.shape[0]
        # D

        self.columns = self.matrix.shape[1]
        self.__fit_em()


    def __fit_em(self, maxit=20):
        self.do_plot(self.matrix,"Initial matrix")

        # W is a DxM array
        w_old = np.random.rand(self.columns, self.final_dimensions)

        # Initialize array Dx1
        hat_x_array = np.empty([self.columns, 1])
        print("rows", self.rows)
        # Construct the array with the mean of each row
        for data_line in range(0, self.columns):
            #print("test %s", np.mean(self.matrix[data_line]))
            hat_x_array[data_line, :] = np.mean(self.matrix[data_line])

        print("Hat x array shape %s", hat_x_array.shape)
        # print("Hat x array %s", hat_x_array)
        # Copy data array
        normalized_x = self.matrix

        print("Normalized_x array size ", normalized_x.shape)
        # print("Normalized_x array ", normalized_x)

        hat_x_array = np.repeat(hat_x_array, self.rows, axis=1)

        # print("Hat x array %s", hat_x_array)

        normalized_x = np.subtract(normalized_x, hat_x_array.T)
        normalized_x = tr(normalized_x)
        print("Normalized_x array after subtraction , shape is", normalized_x.shape)

        # The old sigma
        sigma_old = self.sigma_old

        # The sigma square
        sigma_square = self.sigma**2

        M = tr(w_old).dot(w_old)+sigma_square*np.eye(self.final_dimensions)


        for iteration in range(maxit):
            print("M ", inv(M).shape)

            print("W old ",w_old.shape)
            print("Normalized ", normalized_x.shape)
            print("++++++++")

            E_z = inv(M).dot(tr(w_old)).dot(normalized_x)
            E_z_z_transpose = sigma_square * inv(M) + E_z.dot(tr(E_z))

            w_new_left = 0
            w_new_right = 0
            sigma_square_new = 0

            #print("E z%s", E_z.shape)
            print("E z z transpose  %s", E_z_z_transpose.shape)

            print("++++++++", self.columns)

            for n in range(0, self.rows):
                #print("Normalized x is  and shape is ", normalized_x[:, n].reshape(self.columns, 1).shape)
                w_new_left += normalized_x[:,n].reshape(self.columns, 1).dot(tr(E_z[:, n].reshape(self.columns, 1)))

            print("w_new_left---> ",w_new_left.shape)

            w_new_right = E_z_z_transpose ** -1
            print("w_new_right---> ", w_new_right.shape)
            w_new = w_new_left * w_new_right
            #print("w new ->", w_new)

            print("w new shape ->", w_new.shape)
            print("ez z tra shape", E_z_z_transpose.shape)
            print("columns", self.columns)
            print("normalized" , normalized_x.shape)
            print("E z shape", E_z.shape)
            print("rows", self.rows)

            for n in range(0, self.rows):
                sigma_square_new += np.linalg.norm(normalized_x[:, n].reshape(1, self.columns)) ** 2 \
                                    - 2 * tr(E_z[:, n].reshape(self.final_dimensions,1))\
                                    .dot(tr(w_new).dot(normalized_x[:, n].reshape(self.columns,1)))

            sigma_square_new += np.trace(E_z_z_transpose.dot(tr(w_new).dot(w_new)))
            print("sigma new shape ->", sigma_square_new.shape)
            print("sigma shape ->", sigma_square_new)
            sigma_square = sigma_square_new * 1/(self.final_dimensions * self.rows)

            w_old = w_new
            print("w new --> %s",w_new.shape)

        print(self.final_dimensions, self.rows, self.columns)
        U, S ,V = np.linalg.svd(w_old,full_matrices=False)
        self.do_plot((tr(U).dot(normalized_x)).T, "New matrix")

        return

    def do_plot(self, data, title):
        print("data space", data.shape)
        plt.scatter(data[y == 0, 0], data[y == 0, 1], color='red', marker='^', alpha=0.5, label='Circle_01')
        plt.scatter(data[y == 1, 0], data[y == 1, 1], color='blue', marker='o', alpha=0.5, label='Circle_02')
        plt.grid(True)
        plt.xlabel('Pca_01')
        plt.ylabel('Pca_02')
        plt.legend(numpoints=1, loc='lower right')
        plt.title('Projection')
        plt.savefig("Theoretical_03.png")
        plt.show()