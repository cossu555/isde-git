from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Function to load
    Parameters
    ----------
    filename

    Returns
    -------

    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass
