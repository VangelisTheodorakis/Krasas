import matplotlib.pyplot as plt
import numpy as np
from numpy import transpose as tr

from sklearn.datasets import make_circles
from sklearn.decomposition import PCA

X, y = make_circles(n_samples=1000, factor=.3, noise=.05)

"""
X array is D x N

"""


def do_PCA(x_array=X):
    do_plot(x_array, "Dataset")
    hat_x_array = np.empty([x_array.shape[0], x_array.shape[1]])

    for data_column in range(0, x_array.shape[1]):
        print("peos")


        print("column ", x_array[:, data_column])
        print("mean ", np.mean(x_array[:, data_column]))
        hat_x_array[:, data_column] = np.mean(x_array[:, data_column])

    print("hat x is %s ", hat_x_array)
    normalized_x = np.subtract(x_array, hat_x_array)
    print("x array", x_array)
    print("normalized x %s", normalized_x)

    #covarianve_matrix = (1/x_array.shape[1]) * normalized_x.dot(tr(normalized_x))
    covarianve_matrix = np.cov(x_array)

    print("covariance ", covarianve_matrix)
    print("conv ", np.cov(normalized_x))
    eigenvalues, eigenvectors = np.linalg.eig(covarianve_matrix)

    for i in range(len(eigenvalues)):
        eigv = eigenvectors[:, i].T
        np.testing.assert_array_almost_equal(covarianve_matrix.dot(eigv), eigenvalues[i] * eigv,decimal=6, err_msg='', verbose=True)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # sort eigenvectors according to same index
    eigenvalues = eigenvalues[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    print("--->",eigenvectors.shape)
    eigenvectors = eigenvectors[:2, :2]
    print("--->", eigenvectors.shape)
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    
    projections = eigenvectors.dot(tr(x_array))

    print("projection ", projections.shape)

    do_plot(projections.T, "Custom PCA")
    validate(x_array)


def do_plot(data, title):
    print("data space" , data.shape)
    plt.scatter(data[y == 0, 0], data[y == 0, 1], color='red', marker='^', alpha=0.5, label='Circle_01')
    plt.scatter(data[y == 1, 0], data[y == 1, 1], color='blue', marker='o', alpha=0.5, label='Circle_02')
    plt.grid(True)
    plt.xlabel('Pca_01')
    plt.ylabel('Pca_02')
    plt.legend(numpoints=1, loc='lower right')
    plt.title('Projection')
    plt.savefig("Theoretical_01.png")
    plt.show()

    """"
    plt.figure()
    plt.title(title)
    reds = y == 0
    blues = y == 1

    plt.scatter(data[reds, 0], data[reds, 1], "ro")
    plt.scatter(data[blues, 0], data[blues, 1], "bo")
    print(data.shape)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.show()
    """


def validate(x_array):
        pca = PCA()
        standard_pca = pca.fit_transform(x_array)
        do_plot(standard_pca, "Standard PCA")
