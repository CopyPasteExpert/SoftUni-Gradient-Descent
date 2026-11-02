import numpy as np

def quadratic_f(p, kappa):
    """Value of the bowl ½(x² + κy²) at point p."""

    x, y = p[0], p[1]
    return 0.5 * (x**2 + kappa * y**2)


def quadratic_grad(p, kappa):
    """Gradient of the bowl ½(x² + κy²) at point p."""

    x, y = p[0], p[1]
    return np.array([x, kappa * y])


def numerical_gradient(f, p, h = 1e-6):
    """Numerical gradient of function f at point p using finite differences."""

    grad = np.zeros_like(p)
    
    for i in range(len(p)):
        p_plus_h = p.copy()
        p_minus_h = p.copy()
        p_plus_h[i] += h
        p_minus_h[i] -= h
        grad[i] = (f(p_plus_h) - f(p_minus_h)) / (2 * h)
    return grad
