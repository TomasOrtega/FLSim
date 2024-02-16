import numpy as np
from experiment_main import Experiment
from log_reg_utils import get_args_as_obj
import scienceplots
import matplotlib
from matplotlib import pyplot as plt
import csv


RESULTS_FOLDER = "results/QAFeLvsNaiveQSGD"


def plot_from_args(args, loss_values):
    args_obj = get_args_as_obj(args)
    label = None
    if args_obj.algorithm_type == "FedBuff":
        label = "Unquantized"
    else:
        label = f"{args_obj.algorithm_type}, server {int(np.log2(args_obj.server_quantizer_value))}-bit {args_obj.server_quantizer_type}"
    plt.plot(
        [x - args_obj.baseline_loss for x in loss_values],
        label=label,
        linestyle="solid",
    )


def get_losses(args):
    experiment = Experiment(args)
    runname = experiment.runname
    folder = f"{RESULTS_FOLDER}/{runname}"
    try:
        losses = np.load(f"{folder}/loss_values.npy")
        print(f"Loaded losses from {folder}")
    except:
        losses = experiment.run_experiment()
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

args3 = {
    "algorithm_type": "FedBuff",
    "baseline_loss": 0.014484174216922262,
    "client_lr": 2,
    "client_quantizer_type": None,
    "client_quantizer_value": None,
    "n_clients": 100,
    "n_global_steps": 10000,
    "n_local_steps": 10,
    "results_folder": RESULTS_FOLDER,
    "seed": 0,
    "server_buffer_size": 10,
    "server_lr": 0.1,
    "server_quantizer_type": None,
    "server_quantizer_value": None,
    "test_run": False,
    "verbose": False,
}

args = [get_args_as_obj(x) for x in [args1, args2, args3]]

# get losses from folder if it exists
losses = [get_losses(x) for x in args]

plt.style.use(['ieee', 'bright'])
# Avoid Type 3 fonts for IEEE publications (switch to True)
matplotlib.rcParams['text.usetex'] = True

# Plot the results
markers = [',', 'o', '^', '*', 'd', 's', 'X', 'P', '.', 6, 7]
fig = plt.figure()

for (arg, loss) in zip(args, losses):
    plot_from_args(arg, loss)

plt.xlabel("Global model iteration")
plt.ylabel(r"$f(x) - f^*$")
plt.yscale("log")
plt.grid()
plt.legend(loc="upper right")
plt.tight_layout()
fig.savefig(RESULTS_FOLDER + "/QAFeLvsNaiveQSGD.pdf")
