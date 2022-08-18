import numpy as np
import scipy.sparse as sp

def equal_sparse_matrices(a, b):
    return (a.shape == b.shape) and ((a != b).count_nonzero() == 0)


def close_sparse_matrices(a, b):
    return (a.shape == b.shape) and np.all(np.isclose(a.todense(), b.todense()))


def is_zero_sparse_matrix(a):
    return a.count_nonzero() == 0


def sp_from_coordinates(coords, shape):
    coords = np.asarray(coords)
    return sp.csr_matrix((coords[:, 0], (coords[:, 1], coords[:, 2])), shape=shape)
