import numpy as np
import matplotlib.pyplot as plt

def rejection_sampling_uniform(f, xmin, xmax, N,  *args, **kwargs):
    """
    Rejection sampling with a uniform proposal on [xmin, xmax].
    This algorithm is valid iff f(x) < M.
    If we encounter a value of f(x) exceeding the current estimate of M, we reset.

    Parameters
    ----------
    f : callable
        Target density function (proportional) defined on [xmin, xmax].
        Should accept a float or numpy array as first argument, followed by optional arguments.
    xmin : float
        Lower bound of the domain.
    xmax : float
        Upper bound of the domain.
    N : int
        Number of samples to generate.
    *args : tuple, optional
        Additional positional arguments to pass to the target function `f`.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the target function `f`.

    Returns
    -------
    samples : np.ndarray
        Array of length `N` containing the accepted samples.
    M : float
        Final estimate of the maximum value of f(x) encountered during sampling
    """
    
    x0 = np.random.uniform(xmin, xmax)
    M = f(x0, *args, **kwargs)        # First estimate of M
    samples = []

    while len(samples) < N:
        X = np.random.uniform(xmin, xmax) # Proposal
        fX = f(X, *args, **kwargs)
        if fX > M:
            M = fX
            samples = []
            continue

        U = np.random.uniform(0, 1) # Likelihood ratio
        if U <= fX / M:
            samples.append(X)

    return np.array(samples), M

