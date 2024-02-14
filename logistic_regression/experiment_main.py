import csv
import argparse
import os
import sys
import queue
from tqdm import tqdm  # Import tqdm
from sklearn.datasets import load_svmlight_file
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import scipy.optimize
from log_reg_utils import loss, loss_grad, OPTIMAL_WEIGHTS, plot_losses, qsgd, top_k
import yaml
from absl import app
from absl.flags import argparse_flags


# Set the current time for asynchronous training
global current_time
current_time = 0

# Function to model client delays


def get_client_delay():
    return np.abs(np.random.normal())


def local_training(n_local_steps, lr, weights, data_client, labels_client, l2_strength):
    for t in range(n_local_steps):
        # Update the weights using gradient descent
        weights -= lr * loss_grad(weights, data_client,
                                  labels_client, l2_strength)
    return weights


def train_client(priority_queue, weights, client, n_local_steps, lr, data_clients, labels_clients, l2_strength, client_quantizer=None):
    """Train a client on a given number of local steps.	
    """
    original_weights = weights.copy()
    data_client = data_clients[client]
    labels_client = labels_clients[client]
    weights = local_training(n_local_steps, lr, weights,
                             data_client, labels_client, l2_strength)
    # Calculate the difference in weights
    delta_weights = weights - original_weights

    if client_quantizer is not None:
        delta_weights = client_quantizer(delta_weights)

    # Get the client delay
    client_delay = get_client_delay()

    priority_queue.put((current_time + client_delay, client, delta_weights))


def fill_server_buffer(
    priority_queue,
    global_model,
    n_local_steps,
    client_lr,
    server_lr,
    server_quantizer,
    client_quantizer,
    data_clients,
    labels_clients,
    l2_strength,
    server_buffer_size,
    data,
    target,
    hidden_state=None
):

    # Make an auxiliary model
    aux_model = np.zeros_like(global_model)

    # Fill the buffer
    for _ in range(server_buffer_size):
        global current_time

        # Get the next model from the queue
        current_time, client, client_delta = priority_queue.get()

        # Update the auxiliary model with the client model
        aux_model += client_delta

        # Send client back to training
        model_to_send = None
        if hidden_state is not None:
            model_to_send = hidden_state.copy()
        elif server_quantizer is not None:
            model_to_send = server_quantizer(global_model.copy())
        else:
            model_to_send = global_model.copy()

        train_client(
            priority_queue, model_to_send, client, n_local_steps, client_lr, data_clients, labels_clients, l2_strength, client_quantizer
        )

    # Update the global model with the server learning rate
    global_model += server_lr / server_buffer_size * aux_model

    if hidden_state is not None:
        # Update the hidden state with the server learning rate
        hidden_state += server_quantizer((global_model - hidden_state).copy())

    # Return the logistic loss (cost function) for the current weights
    return loss(global_model, data, target, l2_strength)


def get_quantizer(algorithm_type, quantizer_type, quantizer_value, dim):
    if algorithm_type == "FedBuff":
        return None

    if quantizer_type == "qsgd":
        return lambda x: qsgd(x, quantizer_value)

    if quantizer_type == "top_k":
        return lambda x: top_k(x, int(quantizer_value / 100.0 * dim))

    raise ValueError("Invalid quantizer type")


def get_runname(args):
    from log_reg_utils import config_dict_to_str
    algname = args.algorithm_type
    runname = config_dict_to_str(vars(args),
                                 record_keys=('n_local_steps', 'client_quantizer_type', 'client_quantizer_value', 'server_quantizer_type', 'server_quantizer_value'), prefix=algname)
    return runname


def run_experiment(args):
    seed = args.seed
    np.random.seed(seed)

    ##################### BEGIN: Good old bookkeeping #########################
    runname = get_runname(args)
    save_dir = os.path.join(args.results_folder, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # TODO -- use a logger
    with open(os.path.join(save_dir, f"{runname}_args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    print(f"Running experiment {runname}")
    print(f"Saving to {save_dir}")
    ##################### END: Good old bookkeeping #########################

    # Get dataset
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

    # Set the baseline loss
    baseline_loss = args.baseline_loss
    if baseline_loss is None:
        # Use a black-box optimizer to find the baseline loss
        baseline_loss = scipy.optimize.minimize(
            loss, OPTIMAL_WEIGHTS,
            args=(data, target, l2_strength),
            options={"disp": True}
        ).fun

    n_clients = args.n_clients
    assert n_clients > 0, "Number of clients must be positive"

    # Split the dataset into n_clients parts for clients
    data_clients = np.array_split(data, n_clients)
    labels_clients = np.array_split(target, n_clients)

    # Restart time for each experiment -- avoids precision issues
    global current_time
    current_time = 0

    # Initialize the hidden state
    hidden_state = None
    if args.algorithm_type == "QAFeL":
        hidden_state = np.zeros(d + 1)
    elif args.algorithm_type == "FedBuff" or args.algorithm_type == "Naive":
        hidden_state = None
    else:
        raise NotImplementedError("Invalid algorithm type")

    # Use a priority queue to model asynchrony
    priority_queue = queue.PriorityQueue()

    # Define a global model
    global_model = np.zeros(d + 1)

    client_quantizer = get_quantizer(
        args.algorithm_type, args.client_quantizer_type, args.client_quantizer_value, d + 1)
    server_quantizer = get_quantizer(
        args.algorithm_type, args.server_quantizer_type, args.server_quantizer_value, d + 1)

    # Add all clients to the priority queue
    for client in range(args.n_clients):
        train_client(
            priority_queue, global_model.copy(
            ), client, args.n_local_steps, args.client_lr, data_clients, labels_clients, l2_strength, client_quantizer
        )

    # Initialize loss_values
    loss_values = []

    # Use tqdm to create a progress bar
    for t in tqdm(range(args.n_global_steps)):
        loss_val = fill_server_buffer(
            priority_queue,
            global_model,
            args.n_local_steps,
            args.client_lr,
            args.server_lr,
            server_quantizer,
            client_quantizer,
            data_clients,
            labels_clients,
            l2_strength,
            args.server_buffer_size,
            data,
            target,
            hidden_state
        )
        loss_values.append(loss_val)

    if args.test_run:
        print("Test run complete. Exiting.")
        sys.exit(0)
    else:
        save_results(loss_values, save_dir, runname)
        print(f"Results saved to {save_dir}")

    plt.figure()
    plt.plot(
        [loss - baseline_loss for loss in loss_values],
        label=runname,
        marker='o',
        markevery=int(len(loss_values)/10),
        linestyle="solid",
    )
    plt.xlabel("Global model iteration")
    plt.ylabel(r"$f(x) - f^*$")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{save_dir}/{runname}.png")
    print(f"Plot saved to {save_dir}/{runname}.png")
    if args.verbose:
        plt.show()

    return loss_values


def save_results(loss_values, save_dir, runname):
    # Save the loss values to a CSV file
    with open(f"{save_dir}/{runname}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["global_step", "loss"])
        writer.writerow([i, loss_values[i]] for i in range(len(loss_values)))

    # Save the loss values to npy file
    np.save(f"{save_dir}/loss_values.npy", loss_values)


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="Plot progress and metrics after training.")
    parser.add_argument("--seed", type=int, default=0)

    # Specifying dataset
    # TO-DO: Add support for other datasets

    # Experiment specific args
    parser.add_argument("--algorithm_type", type=str, default="QAFeL",
                        help="Type of algorithm to use, one of 'QAFeL|Naive|FedBuff'")
    parser.add_argument("--client_quantizer_type", type=str, default="qsgd",
                        help="Type of quantizer to use for the client, one of 'qsgd|top_k'")
    parser.add_argument("--client_quantizer_value", type=int, default=65536,
                        help="Value of the quantizer to use for the client (levels for qsgd, percentage of coordinates for top_k)")
    parser.add_argument("--server_quantizer_type", type=str, default="qsgd",
                        help="Type of quantizer to use for the server, one of 'qsgd|top_k'")
    parser.add_argument("--server_quantizer_value", type=int, default=65536,
                        help="Value of the quantizer to use for the server (levels for qsgd, percentage of coordinates for top_k)")
    parser.add_argument("--n_clients", type=int, default=100,
                        help="Number of clients to use for the experiment")
    parser.add_argument("--n_local_steps", type=int, default=10,
                        help="Number of local steps to use for the experiment")
    parser.add_argument("--client_lr", type=float, default=2,
                        help="Learning rate for the client")
    parser.add_argument("--n_global_steps", type=int, default=10000,
                        help="Number of global steps to use for the experiment")
    parser.add_argument("--server_buffer_size", type=int, default=10,
                        help="Size of the server buffer to use for the experiment")
    parser.add_argument("--server_lr", type=float, default=0.1,
                        help="Learning rate for the server")
    parser.add_argument("--results_folder", type=str, default="./results",
                        help="Folder to save the results of the experiment")
    parser.add_argument("--baseline_loss", type=float,
                        default=0.014484174216922262, help="Baseline loss for the experiment")
    parser.add_argument("--test_run", default=False, action='store_true',
                        help="Whether to run a test run of the experiment")

    # Parse arguments.
    args = parser.parse_args(argv[1:])

    return args


def main(args):
    loss_values = run_experiment(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
