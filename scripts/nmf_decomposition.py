# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause
import logging
from time import time

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

from src.models import nmf_missing as my_nmf

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (32, 32)
rng = RandomState(0)

def get_toy_dataset():
    base = np.zeros((120, 32, 32))
    base[:, 5:10, 0:15] = 1

    base = base + np.random.rand(120, 32, 32) * 0.1
    positions = np.zeros(120)
    for j in range(60):
        pos = int(j / 4)
        positions[60 + j] = pos
        base[60 + j, 18:21, 10 + pos:18 + pos] = 0.6
        base[60 + j, 22:25, 7 + pos:18 + pos] = 0.8
        base[60 + j, 26:29, 3 + pos:18 + pos] = 1
        base[60 + j, :, 17 + pos:] = np.nan
    return base

def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        #vmax = max(comp.nanmax(), -comp.min())
        plt.imshow(comp.reshape(image_shape))#, cmap=plt.cm.gray)
        plt.axis('off')
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

if __name__ == "__main__":
    base = get_toy_dataset()
    n_components = 2
    rot_base = np.transpose(base, (0, 2, 1)).reshape(-1, 32)
    # not_nan_index = ~np.isnan(rot_base).any(axis=1)
    data = np.nan_to_num(rot_base)  # data_nan[not_nan_index] # (n_samples=3360, n_features=32) (m=3360, n=32)

    W, H, n_iter = my_nmf.non_negative_factorization(data, n_components=n_components, init='nndsvda', tol=5e-3)
    # W (n_samples, n_components) -> (m=3360, r=2) not (n=32, r=2)
    # H (n_components, n_features) -> (r=2, n=32) not (r=2, m=3360)
    plt.plot(H.T)
    plt.show()

