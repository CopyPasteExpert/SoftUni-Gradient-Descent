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

def gradient_descent(grad, x0, step, max_iter=10000, tol=1e-8):
    """Gradient descent optimization algorithm."""

    history = [np.asarray(x0, dtype=float)]

    for i in range(max_iter):
        g = grad(history[-1])
        if np.linalg.norm(g) < tol:
            break
        x_new = history[-1] - step * g
        history.append(x_new)
    return history[-1], np.array(history)

def momentum(grad, x0, step, beta, max_iter=10000, tol=1e-8):
    """Momentum optimization algorithm."""

    history = [np.asarray(x0, dtype=float)]
    
    for i in range(max_iter):
        g = grad(history[-1])
        if np.linalg.norm(g) < tol:
            break
        m = 0
        if len(history) > 1:
            m = beta * (history[-1] - history[-2])
        
        x_new = history[-1] - step * g + m
        history.append(x_new)
    return history[-1], np.array(history)

def quadratic_hess(p, kappa):
    """Hessian of the bowl ½(x² + κy²) at point p."""

    return np.array([[1.0, 0.0], [0.0, kappa]])

def newton_method(grad, hess, x0, max_iter=10000, tol=1e-8):
    """Newton's method optimization algorithm."""

    history = [np.asarray(x0, dtype=float)]
    
    for i in range(max_iter):
        g = grad(history[-1])
        if np.linalg.norm(g) < tol:
            break
        H = hess(history[-1])
        x_new = history[-1] - np.linalg.solve(H, g)
        history.append(x_new)
    return history[-1], np.array(history)