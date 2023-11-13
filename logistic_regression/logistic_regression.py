import csv
import queue
from tqdm import tqdm  # Import tqdm
from sklearn.datasets import load_svmlight_file
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import scipy.optimize
import log_reg_utils
# from numba import njit


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

# Use a black-box optimizer to find the baseline loss, with display set to True to print the convergence log
baseline_loss = scipy.optimize.minimize(
    log_reg_utils.loss, log_reg_utils.OPTIMAL_WEIGHTS,
    args=(data, target, l2_strength),
    options={"disp": True}
).fun

# Save the baseline loss to a CSV file
with open("results/logistic_regression_baseline.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(["baseline_loss"])
    writer.writerow([baseline_loss])

# Number of clients
n_clients = 50

# Split the dataset into n_clients parts for clients
data_clients = np.array_split(data, n_clients)
labels_clients = np.array_split(target, n_clients)

global current_time
current_time = 0


def my_fake_delay():
    return n_clients


delays = [np.abs(np.random.normal()) for _ in range(n_clients)]


# Function to model client delays
def get_client_delay(client=None):
    delay = 0

    if client is not None:
        delay = delays[client]
    else:
        delay = np.abs(np.random.normal())

    return delay
    # return my_fake_delay()

# @njit


def local_training(n_local_steps, lr, weights, data_client, labels_client):
    for t in range(n_local_steps):
        # Update the weights using gradient descent
        weights -= lr * log_reg_utils.loss_grad(weights, data_client,
                                                labels_client, l2_strength)
    return weights

# Function to train a client


def train_client(priority_queue, weights, client, n_local_steps, lr):
    original_weights = weights.copy()
    data_client = data_clients[client]
    labels_client = labels_clients[client]
    weights = local_training(n_local_steps, lr, weights,
                             data_client, labels_client)
    # Calculate the difference in weights
    delta_weights = weights - original_weights

    # Get the client delay
    client_delay = get_client_delay()

    priority_queue.put((current_time + client_delay, client, delta_weights))


def fill_server_buffer(
    priority_queue,
    server_buffer_size,
    global_model,
    n_local_steps,
    client_lr,
    server_lr,
):
    # Make an auxiliary model
    aux_model = np.zeros(d + 1)

    # Fill the buffer
    for _ in range(server_buffer_size):
        global current_time

        # Get the next model from the queue
        current_time, client, client_delta = priority_queue.get()

        # Update the auxiliary model with the client model
        aux_model += client_delta

        # Send client back to training
        train_client(
            priority_queue, global_model.copy(), client, n_local_steps, client_lr
        )

    # Update the global model with the server learning rate
    global_model += server_lr / server_buffer_size * aux_model

    # Return the logistic loss (cost function) for the current weights
    return log_reg_utils.loss(global_model, data, target, l2_strength)


def run_experiment(n_local_steps):
    # Use a priority queue for asynchrony
    priority_queue = queue.PriorityQueue()

    # Define the client learning rate
    client_lr = 0.1

    # Define the number of global training steps
    n_global_steps = 125000

    # Define the server buffer size
    server_buffer_size = 10

    # Define a global model
    global_model = np.zeros(d + 1)

    # Add all clients to the priority queue
    for client in range(n_clients):
        train_client(
            priority_queue, global_model.copy(), client, n_local_steps, client_lr
        )

    # Define a server learning rate
    server_lr = 0.5

    # Initialize loss_values
    loss_values = []

    # Use tqdm to create a progress bar
    for t in tqdm(range(n_global_steps)):
        loss = fill_server_buffer(
            priority_queue,
            server_buffer_size,
            global_model,
            n_local_steps,
            client_lr,
            server_lr,
        )
        loss_values.append(loss)

    return loss_values


# Run the experiment for different values of n_local_steps
local_steps_values = [1, 2, 4, 8]
loss_values = []
with open("results/logistic_regression.csv", "w", newline="") as csvfile:
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

plt.savefig("results/logistic_regression.png")
# plt.show()
