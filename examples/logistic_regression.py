import queue
from tqdm import tqdm  # Import tqdm
from sklearn.datasets import load_svmlight_file
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
import urllib.request


def loss(weights, X, y, reg):
    raw_scores = -y * np.dot(X, weights).squeeze()
    baseline_loss = np.mean(np.logaddexp(0, raw_scores))
    baseline_loss += reg * np.sum(np.square(weights)) / 2
    return baseline_loss


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
n_clients = 50

# Split the dataset into n_clients parts for clients
data_clients = np.array_split(data, n_clients)
labels_clients = np.array_split(target, n_clients)

global my_counter
my_counter = 0


def my_fake_delay():
    global my_counter
    my_counter += 1
    return my_counter


# Function to model client delays
def get_client_delay():
    return np.abs(np.random.normal())
    # return my_fake_delay()


# Define a function for training a client
def train_client(priority_queue, weights, client, n_local_steps, lr):
    original_weights = weights.copy()
    data_client = data_clients[client]
    labels_client = labels_clients[client]
    for _ in range(n_local_steps):  # Train for a fixed number of iterations
        # Calculate the logits (raw model output)
        logits = np.dot(data_client, weights)

        # Calculate the probabilities using the logistic function
        probabilities = 1 / (1 + np.exp(-logits))

        # Calculate the gradient of the cost function with L2 regularization
        gradient = np.dot(data_client.T, (probabilities - labels_client)) / len(
            labels_client
        )

        # Add the L2 regularization term
        gradient += l2_strength * weights

        # Update the weights using gradient descent
        weights -= lr * gradient

    # Calculate the difference in weights
    delta_weights = weights - original_weights

    priority_queue.put((get_client_delay(), client, delta_weights))


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
        # Get the next model from the queue
        _, client, client_delta = priority_queue.get()

        # Update the auxiliary model with the client model
        aux_model += client_delta

        # Send client back to training
        train_client(
            priority_queue, global_model.copy(), client, n_local_steps, client_lr
        )

    # Update the global model with the server learning rate
    global_model += server_lr / server_buffer_size * aux_model

    # Return the logistic loss (cost function) for the current weights
    return loss(global_model, data, target, l2_strength)


def run_experiment(n_local_steps):
    # Use a priority queue for asynchrony
    priority_queue = queue.PriorityQueue()

    # Define the client learning rate
    client_lr = 0.01

    # Define the number of global training steps
    n_global_steps = 500

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
    server_lr = 0.01

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
local_steps_values = [1, 4, 16]
loss_values = []
for local_steps in local_steps_values:
    # Fix seed for reproducibility
    np.random.seed(0)
    loss_values.append(run_experiment(local_steps))


# Export the experiment results to a CSV file
import csv

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
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for i in range(len(local_steps_values)):
    plt.plot(
        loss_values[i] - baseline_loss,
        label=f"n_local_steps={local_steps_values[i]}",
    )
plt.xlabel("Global round")
plt.ylabel("Loss suboptimality")
plt.yscale("log")
plt.legend()
plt.savefig("logistic_regression_averaging.png")
