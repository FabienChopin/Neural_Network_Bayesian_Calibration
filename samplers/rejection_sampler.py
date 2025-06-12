import numpy as np
import matplotlib.pyplot as plt

def rejection_sampling_uniform(f, xmin, xmax, N):
    """
    Rejection sampling with a uniform proposal on [xmin, xmax].
    This algorithm is valid iff f(x) < M.
    If we encounter a value of f(x) exceeding the current estimate of M, we reset.

    Parameters:
    - f     : target density function (proportional), defined on [xmin, xmax]
    - xmin  : lower bound of the domain
    - xmax  : upper bound of the domain
    - N     : number of samples desired

    Returns:
    - samples : numpy array of length N containing the accepted samples
    """
    
    x0 = np.random.uniform(xmin, xmax)
    M = f(x0)        # First estimate of M
    samples = []

    while len(samples) < N:
        X = np.random.uniform(xmin, xmax) # Proposal
        fX = f(X)
        if f(X) > M:
            M = fX
            samples = []
            continue

        U = np.random.uniform(0, 1) # Likelihood ratio
        if U <= fX:
            samples.append(X)

    return np.array(samples)

