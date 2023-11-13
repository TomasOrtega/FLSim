import csv
import queue
from tqdm import tqdm  # Import tqdm
from sklearn.datasets import load_svmlight_file
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
import urllib.request
from sklearn.metrics import log_loss
from scipy.special import expit as sigmoid


def loss(weights, X, y, reg):
    probabilities = sigmoid(np.dot(X, weights))
    res = log_loss(y, probabilities)
    res += reg * np.square(weights).sum() / 2
    return res


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

# Create a Logistic Regression model with L2 penalty (ridge) and the specified regularization strength
logistic_regression_model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=l2_strength,
    max_iter=50000,
    tol=0,
)

# Compute baseline
# Only fit the model once, as cost doesn't change over iterations
logistic_regression_model.fit(data, target)

# Calculate the logistic loss (cross-entropy loss)
baseline_loss = loss(
    logistic_regression_model.coef_.transpose(), data, target, l2_strength
)

# Add a bias term (intercept) to the data
data = np.hstack((np.ones((n, 1)), data))

# Initialize the logistic regression weights
weights = np.zeros(d + 1)

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
        # Calculate the probabilities using the sigmoid
        probabilities = sigmoid(np.dot(data_client, weights))

        # Calculate the gradient of the cost function with L2 regularization
        gradient = np.dot(data_client.T, (probabilities - labels_client)) / len(
            labels_client
        )

        # Add the L2 regularization term
        gradient += l2_strength * weights

        # Update the weights using gradient descent
        weights -= lr * gradient

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
    return (loss(global_model, data, target, l2_strength), global_model)


def run_experiment(n_local_steps):
    # Define the client learning rate
    client_lr = 0.1

    # Define the number of global training steps
    n_global_steps = 1000

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


# Run the experiment for different values of n_local_steps
local_steps_values = [1, 16, 32]
loss_values = []
for local_steps in local_steps_values:
    # Fix seed for reproducibility
    np.random.seed(0)
    loss_values.append(run_experiment(local_steps))


# Export the experiment results to a CSV file

with open("logistic_regression_averaging.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(["n_local_steps", "global_step", "loss"])
    for i in range(len(local_steps_values)):
        for j in range(len(loss_values[i])):
            writer.writerow([local_steps_values[i], j, loss_values[i][j]])

# Also save the baseline loss
with open("logistic_regression_averaging_baseline.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(["baseline_loss"])
    writer.writerow([baseline_loss])

# Plot the results

markers = ["o", "s", "^"]
plt.figure(figsize=(12, 8))
for i in range(len(local_steps_values)):
    plt.plot(
        loss_values[i] - baseline_loss,
        label=f"n_local_steps={local_steps_values[i]}",
        marker=markers[i],
        markevery=int(len(loss_values[i])/10),
        linestyle="solid",
    )
plt.xlabel("Global round")
plt.ylabel("Loss suboptimality")
plt.yscale("log")
plt.legend()
plt.savefig("logistic_regression_averaging.png")
