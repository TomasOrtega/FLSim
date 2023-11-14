import scienceplots
import matplotlib
from matplotlib import pyplot as plt
import csv
import numpy as np

filename = "results/logistic_regression.csv"
# filename = "results/logistic_regression_averaging.csv"
baseline_filename = "results/logistic_regression_baseline.csv"

n_local_steps = []
local_steps_values = []
loss_values = []
# Import the experiment results from a CSV file
with open(filename, newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)  # Skip the header
    for row in reader:
        n_local_steps = int(row[0])
        global_step = int(row[1])
        loss = float(row[2])
        if n_local_steps not in local_steps_values:
            local_steps_values.append(n_local_steps)
            loss_values.append([loss])
        else:
            loss_values[-1].append(loss)

baseline_loss = 0
# Import the baseline loss
with open(baseline_filename, newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)  # Skip the header
    baseline_loss = float(next(reader)[0])

plt.style.use(['ieee', 'bright'])
# Avoid Type 3 fonts for IEEE publications (switch to True)
matplotlib.rcParams['text.usetex'] = True

# Get the minimum loss, written to be able to plot during experiment run
min_losses = np.min([min(loss_values[i] for i in range(len(loss_values)))])
if min_losses < baseline_loss:
    print("WARNING: The baseline loss is higher than the minimum loss.")
    baseline_loss = min_losses

# Plot the results
markers = [',', 'o', '^', '*', 'd', 's', 'X', 'P', '.', 6, 7]
fig = plt.figure()
last_index = 65000
for i in range(1, len(local_steps_values)):
    plt.plot(
        np.array(loss_values[i][:last_index]) - baseline_loss,
        label=f"{local_steps_values[i]} local steps",
        # marker=markers[i],
        # markevery=int(len(loss_values[i])/10),
        linestyle="solid",
    )
plt.xlabel("Global model iteration")
plt.ylabel(r"$f(x) - f^*$")
plt.yscale("log")
plt.legend(loc="upper right")
plt.tight_layout()
fig.savefig(filename[:-4] + ".pdf")
