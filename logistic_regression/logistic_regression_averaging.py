import csv
from tqdm import tqdm  # Import tqdm
from sklearn.datasets import load_svmlight_file
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import urllib.request
from sklearn.metrics import log_loss
from scipy.special import expit as sigmoid
import log_reg_utils




# Set the random seed for reproducible results
np.random.seed(0)

# Download the LIBSVM mushroom dataset from the URL
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
urllib.request.urlretrieve(url, "mushroom.libsvm")

# Load the downloaded dataset
data, target = load_svmlight_file("mushroom.libsvm")

# Convert the sparse data to a dense matrix
data = data.toarray()

# Bring target to 0,1
target = target - 1

# Get problem dimensions
n, d = data.shape

# Set the L2 regularization strength
l2_strength = 1.0 / n

# Add a bias term (intercept) to the data
data = np.hstack((np.ones((n, 1)), data))

# Initialize the logistic regression weights
weights = np.zeros(d + 1)

# Initial guess for the optimizer
initial_weights = log_reg_utils.OPTIMAL_WEIGHTS

# Use a black-box optimizer to find the baseline loss, with display set to True to print the convergence log
baseline_loss = scipy.optimize.minimize(
    log_reg_utils.loss, initial_weights,
    args=(data, target, l2_strength),
    jac=log_reg_utils.loss_grad,
    tol=0,
    options={"disp": True}
).fun

# Save the baseline loss to a CSV file
with open("results/logistic_regression_baseline.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(["baseline_loss"])
    writer.writerow([baseline_loss])

# Number of clients
n_clients = 1

# Split the dataset into n_clients parts for clients
data_clients = np.array_split(data, n_clients)
labels_clients = np.array_split(target, n_clients)


# Define a function for training a client
def train_client(weights, client, n_local_steps, lr):
    data_client = data_clients[client]
    labels_client = labels_clients[client]
    for _ in range(n_local_steps):  # Train for a fixed number of iterations
        # Update the weights using gradient descent
        weights -= lr * log_reg_utils.loss_grad(weights, data_client,
                                  labels_client, l2_strength)

    return weights


def fill_server_buffer(
    global_model,
    n_local_steps,
    client_lr,
):
    # Make an auxiliary model
    aux_model = np.zeros(d + 1)

    # Fill the buffer
    for client in range(n_clients):
        # Update the auxiliary model with the client model
        aux_model += train_client(global_model.copy(),
                                  client, n_local_steps, client_lr)

    # Update the global model with the server learning rate
    global_model = aux_model / n_clients

    # Return the logistic loss (cost function) for the current weights
    return (log_reg_utils.loss(global_model, data, target, l2_strength), global_model)


def run_experiment(n_local_steps):
    # Define the client learning rate
    client_lr = 1

    # Define the number of global training steps
    n_global_steps = 25000

    # Define a global model
    global_model = np.zeros(d + 1)

    # Initialize loss_values
    loss_values = []

    # Use tqdm to create a progress bar
    for t in tqdm(np.arange(n_global_steps)):
        loss, global_model = fill_server_buffer(
            global_model,
            n_local_steps,
            client_lr,
        )
        loss_values.append(loss)

    return loss_values


# Run the experiment for different values of n_local_steps, while saving to a CSV file
local_steps_values = [1, 4, 16]
loss_values = []
with open("results/logistic_regression_averaging.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(["n_local_steps", "global_step", "loss"])
    for local_steps in local_steps_values:
        # Fix seed for reproducibility
        np.random.seed(0)
        loss_values.append(run_experiment(local_steps))
        [writer.writerow([local_steps, i, loss_values[-1][i]])
         for i in range(len(loss_values[-1]))]

# Plot the results

fig = log_reg_utils.plot_losses(local_steps_values, loss_values, baseline_loss)
plt.savefig("results/logistic_regression_averaging.png")
