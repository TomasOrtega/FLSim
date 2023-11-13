import scienceplots
import matplotlib
from matplotlib import pyplot as plt
import csv
import numpy as np


n_local_steps = []
local_steps_values = []
loss_values = []
# Import the experiment results from a CSV file
with open("logistic_regression.csv", newline="") as csvfile:
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
with open("logistic_regression_baseline.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader)  # Skip the header
    baseline_loss = float(next(reader)[0])

plt.style.use('ieee')
# Avoid Type 3 fonts for IEEE publications (switch to True)
matplotlib.rcParams['text.usetex'] = True

# Plot the results

markers = ["o", "s", "^"]
fig = plt.figure()
for i in range(len(local_steps_values)):
    plt.plot(
        loss_values[i] - baseline_loss,
        label=f"{local_steps_values[i]} local steps",
        marker=markers[i],
        markevery=int(len(loss_values[i])/10),
        linestyle="solid",
    )
plt.xlabel("Global model iteration")
plt.ylabel(r"$f(x) - f^*$")
plt.yscale("log")
plt.legend()
plt.tight_layout()
fig.savefig("logistic_regression.pdf")
