import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression


def mutual_information_sklearn(x, y):
    return mutual_info_regression(x, y)[0]


def compute_histogram_counts(x, bins):
    """
    Computes the histogram counts for a variable.

    Parameters:
    - x: array-like, shape (n_samples,)
        Variable data.
    - bins: int or sequence of scalars
        Number of bins or bin edges.

    Returns:
    - counts: ndarray
        Counts in each bin.
    - bin_edges: ndarray
        Bin edges.
    """
    counts, bin_edges = np.histogram(x, bins=bins, density=False)
    return counts, bin_edges

def compute_joint_histogram_counts(x, y, bins_x, bins_y):
    """
    Computes the joint histogram counts for two variables.

    Parameters:
    - x: array-like, shape (n_samples,)
        First variable data.
    - y: array-like, shape (n_samples,)
        Second variable data.
    - bins_x: sequence of scalars
        Bin edges for x.
    - bins_y: sequence of scalars
        Bin edges for y.

    Returns:
    - counts_xy: ndarray
        Joint counts.
    """
    counts_xy, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], density=False)
    return counts_xy

def compute_entropy_from_counts(counts, base=2):
    """
    Computes the entropy from counts.

    Parameters:
    - counts: array-like
        Counts for each bin or event.
    - base: float, optional
        Logarithm base (default is 2 for bits).

    Returns:
    - H: float
        Entropy.
    """
    counts = np.asarray(counts)
    total = counts.sum()
    probabilities = counts / total
    # Remove zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    H = entropy(probabilities, base=base)
    return H

def compute_entropies(x, y, bins=None):
    """
    Computes the joint entropy H(X, Y) and marginal entropies H(X), H(Y) of two variables.

    Parameters:
    - x: array-like, shape (n_samples,)
        First variable (can be continuous or discrete).
    - y: array-like, shape (n_samples,)
        Second variable (can be continuous or discrete).
    - bins: int or sequence of ints or None (optional)
        Number of bins for discretizing continuous variables.
        If None, defaults to the square root of the number of samples.

    Returns:
    - H_X: float
        Entropy of X.
    - H_Y: float
        Entropy of Y.
    - H_XY: float
        Joint entropy of X and Y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Check that x and y have the same length
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    n_samples = x.shape[0]

    # If bins is None, choose default number of bins
    if bins is None:
        bins = int(np.sqrt(n_samples))

    # Compute counts and bin edges for x and y
    counts_x, bin_edges_x = compute_histogram_counts(x, bins)
    counts_y, bin_edges_y = compute_histogram_counts(y, bins)

    # Compute joint counts
    counts_xy = compute_joint_histogram_counts(x, y, bin_edges_x, bin_edges_y)

    # Compute entropies from counts
    H_X = compute_entropy_from_counts(counts_x)
    H_Y = compute_entropy_from_counts(counts_y)
    H_XY = compute_entropy_from_counts(counts_xy.flatten())

    return H_X, H_Y, H_XY

def normalized_mutual_information(x, y, bins=None):
    """
    Computes the normalized mutual information (NMI) between two variables x and y.

    The NMI is calculated using the entropies of x, y, and their joint entropy:
    NMI = 2 * (H(X) + H(Y) - H(X,Y)) / (H(X) + H(Y))

    Parameters:
    - x: array-like, shape (n_samples,)
        First variable (can be continuous or discrete).
    - y: array-like, shape (n_samples,)
        Second variable (can be continuous or discrete).
    - bins: int or sequence of ints or None (optional)
        Number of bins for discretizing continuous variables.
        If None, defaults to the square root of the number of samples.

    Returns:
    - NMI: float
        Normalized Mutual Information between x and y.
    """
    H_X, H_Y, H_XY = compute_entropies(x, y, bins)
    numerator = 2 * (H_X + H_Y - H_XY)
    denominator = H_X + H_Y
    if denominator == 0:
        NMI = 0.0  # Avoid division by zero; define NMI as zero when entropies sum to zero
    else:
        NMI = numerator / denominator
    return NMI
