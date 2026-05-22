import numpy as np

def transform(x, lambda1=1/3, lambda2=1):
    if np.any(x <= -lambda2):
        raise ValueError("All values must be greater than -lambda2")

    if lambda1 == 0:
        x_t = np.ln(x + lambda2)
    else:
        x_t = ((x + lambda2) ** lambda1 - 1) / lambda1

    return x_t


def inverse(x_t, lambda1=1/3, lambda2=1):
    if lambda1 == 0:
        x = np.exp(x_t) - lambda2
    else:
        x = (lambda1 * x_t + 1) ** (1 / lambda1) - lambda2

    return x