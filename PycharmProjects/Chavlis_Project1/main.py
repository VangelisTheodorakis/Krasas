import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
from ppca import PPCA


def plot_scatter(x, classes, ax=None):
    ax = plt.gca() if ax is None else ax
    cmap = 'jet'
    norm = plt_col.Normalize(vmin=np.min(classes), vmax=np.max(classes))
    mapper = plt_cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(classes)
    ax.scatter(x[0, :], x[1, :], color=colors, s=20)
    plt.show()

# test
iris = ds.load_iris()
iris_y = np.transpose(iris.data)

iris_classes = iris.target

# PPCA
magic = PPCA()
magic.fit(iris_y)

plot_scatter(magic.transform(), iris_classes)
