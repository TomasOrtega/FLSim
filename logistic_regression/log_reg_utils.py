
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


def plot_losses(legend_values, loss_values, baseline_loss, label="local steps"):
    # Plot the results
    markers = [',', 'o', '^', '*', 'd', 's', 'X', 'P', '.', 6, 7]
    fig = plt.figure()
    for i in range(len(legend_values)):
        plt.plot(
            np.array(loss_values[i]) - baseline_loss,
            label=f"{legend_values[i]} {label}",
            marker=markers[i],
            markevery=int(len(loss_values[i])/10),
            linestyle="solid",
        )
    plt.xlabel("Global model iteration")
    plt.ylabel(r"$f(x) - f^*$")
    plt.yscale("log")
    plt.legend()
    return fig


def qsgd(input_vector, num_levels, norm_threshold=1e-10):
    """
    Quantize the input vector with a specified number of levels using QSGD.

    Parameters:
    - input_vector (numpy.ndarray): Input vector to be quantized.
    - num_levels (int): Number of quantization levels.
    - norm_threshold (float, optional): Small norm threshold. If the norm of the input_vector is below this threshold, return zero.

    Returns:
    - numpy.ndarray: Quantized vector.

    The QSGD (Quantized Stochastic Gradient Descent) quantizer works by dividing the vector into quantization levels
    and randomly deciding whether to round up to the next level based on the normalized magnitude of each element.

    If the norm of the input vector is below the specified threshold, the function returns a zero vector.
    """
    norm_input_vector = np.linalg.norm(input_vector)

    # Check if the norm is small, return zero
    if norm_input_vector < norm_threshold:
        return np.zeros_like(input_vector)

    level_float = num_levels * np.abs(input_vector) / norm_input_vector
    level_floor = np.floor(level_float)
    is_next_level = np.random.rand(
        input_vector.shape[0]) < (level_float - level_floor)
    level = level_floor + is_next_level

    return np.sign(input_vector) * norm_input_vector * level / num_levels


def top_k(input_vector, k):
    """
    Returns a vector with the top k elements of the input vector, setting all other elements to zero.

    Parameters:
    - input_vector (numpy.ndarray): Input vector.
    - k (int): Number of top elements to retain.

    Returns:
    - numpy.ndarray: A vector where only the top k elements of the input vector are preserved.
    """
    topk_indices = np.argsort(np.abs(input_vector))[-k:]
    output_vector = np.zeros_like(input_vector)
    output_vector[topk_indices] = input_vector[topk_indices]
    return output_vector


cmdline_arg_abbr = {
    'n_local_steps': 'ls',
    'client_quantizer_type': 'c',
    'client_quantizer_value': '',
    'server_quantizer_type': 's',
    'server_quantizer_value': '',
}


def config_dict_to_str(args_dict, record_keys=tuple(), leave_out_falsy=True, prefix=None, use_abbr=True,
                       primary_delimiter='-', secondary_delimiter='_'):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    :param args_dict:
    :param record_keys: a tuple/list of keys that is a subset of keys in args_dict that will be used to form the runname
    :param leave_out_falsy: whether to skip keys whose values evaluate to falsy (0, None, False, etc.)
    :param use_abbr: whether to use abbreviations for long key name
    :param primary_delimiter: the char to delimit different key-value paris
    :param secondary_delimiter: the delimiter within each key or value string (e.g., when the value is a list of numbers)
    :return:
    """
    kv_strs = []  # ['key1=val1', 'key2=val2', ...]

    for key in record_keys:
        val = args_dict[key]
        if leave_out_falsy and not val:
            continue
        # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
        if isinstance(val, (list, tuple)):
            val_str = secondary_delimiter.join(map(str, val))
        else:
            val_str = str(val)
        if use_abbr:
            key = cmdline_arg_abbr.get(key, key)
        kv_strs.append('%s=%s' % (key, val_str))

    if prefix:
        substrs = [prefix] + kv_strs
    else:
        substrs = kv_strs
    return primary_delimiter.join(substrs)
