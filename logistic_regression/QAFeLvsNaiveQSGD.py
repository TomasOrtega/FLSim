import numpy as np
from experiment_main import run_experiment, get_runname
from log_reg_utils import get_args_as_obj
import scienceplots
import matplotlib
from matplotlib import pyplot as plt
import csv


RESULTS_FOLDER = "results/QAFeLvsNaiveQSGD"


def plot_from_args(args, loss_values):
    args_obj = get_args_as_obj(args)
    plt.plot(
        [x - args_obj.baseline_loss for x in loss_values],
        label=f"{args_obj.algorithm_type}, server {int(np.log2(args_obj.server_quantizer_value))}-bit {args_obj.server_quantizer_type}",
        linestyle="solid",
    )


def get_losses(args):
    runname = get_runname(args)
    folder = f"{RESULTS_FOLDER}/{runname}"
    try:
        losses = np.load(f"{folder}/loss_values.npy")
        print(f"Loaded losses from {folder}")
    except:
        losses = run_experiment(args)
    return losses


args1 = {
    "algorithm_type": "QAFeL",
    "baseline_loss": 0.014484174216922262,
    "client_lr": 2,
    "client_quantizer_type": "qsgd",
    "client_quantizer_value": 4,
    "n_clients": 100,
    "n_global_steps": 10000,
    "n_local_steps": 10,
    "results_folder": RESULTS_FOLDER,
    "seed": 0,
    "server_buffer_size": 10,
    "server_lr": 0.1,
    "server_quantizer_type": "qsgd",
    "server_quantizer_value": 8,
    "test_run": False,
    "verbose": False,
}

args2 = {
    "algorithm_type": "Naive",
    "baseline_loss": 0.014484174216922262,
    "client_lr": 2,
    "client_quantizer_type": "qsgd",
    "client_quantizer_value": 4,
    "n_clients": 100,
    "n_global_steps": 10000,
    "n_local_steps": 10,
    "results_folder": RESULTS_FOLDER,
    "seed": 0,
    "server_buffer_size": 10,
    "server_lr": 0.1,
    "server_quantizer_type": "qsgd",
    "server_quantizer_value": 8,
    "test_run": False,
    "verbose": False,
}

args1 = get_args_as_obj(args1)
args2 = get_args_as_obj(args2)

# get losses from folder if it exists
losses1 = get_losses(args1)
losses2 = get_losses(args2)

plt.style.use(['ieee', 'bright'])
# Avoid Type 3 fonts for IEEE publications (switch to True)
matplotlib.rcParams['text.usetex'] = True

# Plot the results
markers = [',', 'o', '^', '*', 'd', 's', 'X', 'P', '.', 6, 7]
fig = plt.figure()

plot_from_args(args1, losses1)
plot_from_args(args2, losses2)

plt.xlabel("Global model iteration")
plt.ylabel(r"$f(x) - f^*$")
plt.yscale("log")
plt.grid()
plt.legend(loc="upper right")
plt.tight_layout()
fig.savefig(RESULTS_FOLDER + "/QAFeLvsNaiveQSGD.pdf")
