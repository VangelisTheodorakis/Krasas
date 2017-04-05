import pylab as plt
import numpy as np
plt.ion()


def my_kmeans(X, k, distance, maxiter, reps):
    centroids = X[np.random.choice(np.arange(len(X)), k), :]
    for i in range(maxiter):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i - y_k, x_i - y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis=0) for k in range(k)]
    return np.array(centroids), C



def show(X, C, centroids):
    plt.cla()
    plt.plot(X[C == 0, 0], X[C == 0, 1], 'ob',
         X[C == 1, 0], X[C == 1, 1], 'or')
    plt.plot( centroids[:,0], centroids[:,1], '*m', markersize=20)
    plt.draw()
    plt.ioff()
    plt.show()