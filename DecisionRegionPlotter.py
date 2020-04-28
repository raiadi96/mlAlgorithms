#simple contourf + scatter mapping for data with 2 features

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):

    markers = ("s", "x", "o", "v", "^", "p", "P", "*", "h", "+", "X", "D")
    colors = ("violet", "indigo", "blue", "green", "yellow", "orange", "red", "black", "grey", "lightgreen", "cyan" )
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max,resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()].T))
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha = 0.2, cmap = cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl,0], y = X[y==cl,1], alpha=0.8, c = colors[idx], marker= markers[idx], label = cl, edgecolors= 'black')

    if test_idx:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(x = X_test[:,0], y = X_test[:,1], edgecolors= 'white', c = 'white', alpha = 1.0, marker= "o", linewidth =1, label = "test case")

