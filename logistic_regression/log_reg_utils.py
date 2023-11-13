
import numpy as np
import matplotlib.pyplot as plt
# from numba import njit

OPTIMAL_WEIGHTS = np.array([-1.56078757e-01, -5.28758919e-01, -2.67212685e-01, -3.98851510e-02,
                            2.99904413e-02, 5.66867597e-01, 8.29199598e-02, 8.26821995e-02,
                            -1.58834746e-01, -7.94720708e-01, 7.14794498e-01, 7.39326455e-01,
                            -1.05255163e+00, 2.77065048e-03, -1.24319156e-01, 4.33367502e-01,
                            -1.02782215e+00, 4.92888021e-01, 4.92888021e-01, -4.79616903e-01,
                            3.66990429e-01, -6.11513354e-01, 4.55434597e-01, 3.50957561e+00,
                            -3.67421522e+00, -3.58118939e+00, -3.55636703e-01, 3.50957561e+00,
                            4.08280596e+00, -2.06563080e+00, -7.90681910e-01, -7.90681910e-01,
                            2.21622511e-01, -3.77701268e-01, -1.14806284e+00, 9.91984086e-01,
                            2.17962094e+00, -2.33569969e+00, -1.63004891e+00, 6.11330483e-01,
                            -1.89931529e-02, -1.60717840e-01, 5.26999701e-01, 1.51726199e-01,
                            6.12435409e-01, 3.72263893e-01, -8.04022443e-01, 1.53410858e-01,
                            1.70577484e-01, -1.41040440e-01, -5.02018695e-01, 3.45939938e-01,
                            -1.75685757e-01, 1.05301682e+00, -1.47126320e+00, 4.37853384e-01,
                            -1.06434970e+00, 1.97296983e-01, -3.90359913e-01, 1.10133388e+00,
                            -3.55636703e-01, -3.55805459e-01, 5.96954074e-01, 3.37007080e-01,
                            4.50666858e-01, -1.56707724e-01, -3.36445307e-01, 1.16896902e-01,
                            -4.53008478e-01, -3.55636703e-01, -3.00235133e-01, 5.74091619e-01,
                            3.31299815e-01, 4.50666858e-01, 3.23655918e-01, -3.25799952e-01,
                            3.08026405e-02, -8.84923819e-01, -1.56078757e-01, -4.53008478e-01,
                            2.25333429e-01, -1.53737137e-01, 2.25333429e-01, 5.78339353e-02,
                            1.41724010e-01, -3.55636703e-01, -5.26471867e-01, -3.55636703e-01,
                            -3.21723334e-01, -1.15258298e+00, 2.20033613e+00, 1.41941780e-01,
                            -1.33216817e+00, 1.71906539e+00, 1.41941780e-01, 1.88778101e+00,
                            -4.50089875e+00, 1.63158746e+00, 1.27289590e-02, 1.41941780e-01,
                            1.38668483e-02, -1.24691224e+00, 1.08820034e+00, -1.94402148e-01,
                            -6.84991020e-01, 8.68159461e-01, -1.13894455e-01, -3.84143439e-01,
                            -5.43048942e-01, -6.64704879e-01, 6.76462790e-02, -7.42263260e-02,
                            1.55629300e+00])

# See Fabian Pedregosa's https://fa.bianp.net/blog/2019/evaluate_logistic/


def logsig(x):
    """Compute log(1 / (1 + exp(-t))) component-wise."""
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def f(x, A, b):
    """Logistic loss, numerically stable implementation.

    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients

    A: array-like, shape (n_samples, n_features)
        Data matrix

    b: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    loss: float
    """
    z = np.dot(A, x)
    b = np.asarray(b)
    return np.mean((1 - b) * z - logsig(z))


def expit_b(x, b):
    """Compute sigmoid(x) - b component-wise."""
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


def f_grad(x, A, b):
    """Computes the gradient of the logistic loss.

    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients

    A: array-like, shape (n_samples, n_features)
        Data matrix

    b: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    grad: array-like, shape (n_features,)    
    """
    z = A.dot(x)
    s = expit_b(z, b)
    return A.T.dot(s) / A.shape[0]


def loss(weights, X, y, reg):
    """"Computes the loss function for regularized logistic regression with l2 penalty.
    """
    res = f(weights, X, y)
    res += 0.5 * reg * np.square(weights).sum()
    return res


def loss_grad(weights, X, y, reg):
    """Computes the gradient of the loss function for regularized logistic regression with l2 penalty.
    """
    res = f_grad(weights, X, y)
    res += reg * weights
    return res

def plot_losses(local_steps_values, loss_values, baseline_loss):
    # Plot the results
    markers = [',', 'o', '^', '*', 'd', 's', 'X', 'P', '.', 6, 7]
    fig = plt.figure()
    for i in range(len(local_steps_values)):
        plt.plot(
            np.array(loss_values[i]) - baseline_loss,
            label=f"{local_steps_values[i]} local steps",
            marker=markers[i],
            markevery=int(len(loss_values[i])/10),
            linestyle="solid",
        )
    plt.xlabel("Global model iteration")
    plt.ylabel(r"$f(x) - f^*$")
    plt.yscale("log")
    plt.legend()
    return fig