import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
from ppca import PPCA
from pca import *


def plot_scatter(x, classes, ax=None):
    ax = plt.gca() if ax is None else ax
    cmap = 'jet'
    norm = plt_col.Normalize(vmin=np.min(classes), vmax=np.max(classes))
    mapper = plt_cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(classes)
    ax.scatter(x[0, :], x[1, :], color=colors, s=20)
    plt.show()

# Iris
iris = ds.load_iris()

# data.T
iris_y = np.transpose(iris.data)

iris_classes = iris.target
# PCA

do_PCA()



# PPCA
#magic = PPCA()

#magic.fit()
""""
plot_scatter(magic.transform(), iris_classes)
"""