import csv
import argparse
import os
import sys
import queue
import sys
import urllib.request
from tqdm import tqdm
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from log_reg_utils import loss, loss_grad, OPTIMAL_WEIGHTS, qsgd, top_k
import yaml
from absl import app
from absl.flags import argparse_flags


class Experiment:
    def __init__(self, args):
        self.args = args
        self.test_run = args.test_run
        self.server_buffer_size = args.server_buffer_size
        self.client_lr = args.client_lr
        self.server_lr = args.server_lr
        self.verbose = args.verbose
        self.n_global_steps = args.n_global_steps
        self.n_local_steps = args.n_local_steps
        self.n_clients = args.n_clients
        assert self.n_clients > 0, "Number of clients must be positive"
        # Use a priority queue to model asynchrony
        self.priority_queue = queue.PriorityQueue()

        # Set the current time for asynchronous training
        self.current_time = 0
        np.random.seed(args.seed)

        ##################### BEGIN: Good old bookkeeping #########################
        self.runname = self.get_runname()
        self.save_dir = os.path.join(args.results_folder, self.runname)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # TODO -- use a logger
        with open(os.path.join(self.save_dir, f"{self.runname}_args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
        print(f"Running experiment {self.runname}")
        print(f"Saving to {self.save_dir}")
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
        self.l2_strength = 1.0 / n

        # Add a bias term (intercept) to the data
        data = np.hstack((np.ones((n, 1)), data))

        # Define a global model
        self.global_model = np.zeros(d + 1)

        # Set the baseline loss
        self.baseline_loss = args.baseline_loss
        if self.baseline_loss is None:
            # Use a black-box optimizer to find the baseline loss
            self.baseline_loss = scipy.optimize.minimize(
                loss, OPTIMAL_WEIGHTS,
                args=(data, target, self.l2_strength),
                options={"disp": True}
            ).fun

        # Split the dataset into n_clients parts for clients
        self.data_clients = np.array_split(data, self.n_clients)
        self.labels_clients = np.array_split(target, self.n_clients)

        # Initialize the hidden state
        self.hidden_state = None
        if args.algorithm_type == "QAFeL":
            self.hidden_state = np.zeros_like(self.global_model)
        elif args.algorithm_type == "FedBuff" or args.algorithm_type == "Naive":
            self.hidden_state = None
        else:
            raise NotImplementedError("Invalid algorithm type")

        self.client_quantizer = self.get_quantizer(
            args.algorithm_type, args.client_quantizer_type, args.client_quantizer_value, d + 1)
        self.server_quantizer = self.get_quantizer(
            args.algorithm_type, args.server_quantizer_type, args.server_quantizer_value, d + 1)

        self.data, self.target = data, target

    def get_client_delay(self):
        return np.abs(np.random.normal())

    def local_training(self, n_local_steps, lr, weights, data_client, labels_client, l2_strength):
        for t in range(n_local_steps):
            # Update the weights using gradient descent
            weights -= lr * \
                loss_grad(weights, data_client, labels_client, l2_strength)
        return weights

    def train_client(self, weights, client, n_local_steps, lr):
        """Train a client on a given number of local steps."""
        original_weights = weights.copy()
        data_client = self.data_clients[client]
        labels_client = self.labels_clients[client]
        weights = self.local_training(
            n_local_steps, lr, weights, data_client, labels_client, self.l2_strength)
        # Calculate the difference in weights
        delta_weights = weights - original_weights

        if self.client_quantizer is not None:
            delta_weights = self.client_quantizer(delta_weights)

        # Get the client delay
        client_delay = self.get_client_delay()

        self.priority_queue.put(
            (self.current_time + client_delay, client, delta_weights))

    def fill_server_buffer(self):
        # Make an auxiliary model
        aux_model = np.zeros_like(self.global_model)

        # Fill the buffer
        for _ in range(self.server_buffer_size):
            # Get the next model from the queue
            self.current_time, client, client_delta = self.priority_queue.get()

            # Update the auxiliary model with the client model
            aux_model += client_delta

            # Send client back to training
            model_to_send = None
            if self.hidden_state is not None:
                model_to_send = self.hidden_state.copy()
            elif self.server_quantizer is not None:
                model_to_send = self.server_quantizer(self.global_model.copy())
            else:
                model_to_send = self.global_model.copy()

            self.train_client(model_to_send, client,
                              self.n_local_steps, self.client_lr)

        # Update the global model with the server learning rate
        self.global_model += self.server_lr / self.server_buffer_size * aux_model

        if self.hidden_state is not None:
            # Update the hidden state with the server learning rate
            self.hidden_state += self.server_quantizer(
                (self.global_model - self.hidden_state).copy())

        # Return the logistic loss (cost function) for the current weights
        return loss(self.global_model, self.data, self.target, self.l2_strength)

    def get_quantizer(self, algorithm_type, quantizer_type, quantizer_value, dim):
        if algorithm_type == "FedBuff":
            return None

        if quantizer_type == "qsgd":
            return lambda x: qsgd(x, quantizer_value)

        if quantizer_type == "top_k":
            return lambda x: top_k(x, int(quantizer_value / 100.0 * dim))

        raise ValueError("Invalid quantizer type")

    def get_runname(self):
        from log_reg_utils import config_dict_to_str
        args = self.args
        algname = args.algorithm_type
        runname = config_dict_to_str(vars(args), record_keys=('n_local_steps', 'client_quantizer_type',
                                     'client_quantizer_value', 'server_quantizer_type', 'server_quantizer_value'), prefix=algname)
        return runname

    def run_experiment(self):
        # Reset the global time
        self.current_time = 0

        # Add all clients to the priority queue
        for client in range(self.n_clients):
            self.train_client(self.global_model.copy(), client,
                              self.n_local_steps, self.client_lr)

        # Initialize loss_values
        loss_values = []

        # Use tqdm to create a progress bar
        for t in tqdm(range(self.n_global_steps)):
            loss_val = self.fill_server_buffer()
            loss_values.append(loss_val)

        if self.test_run:
            print("Test run complete. Exiting.")
            sys.exit(0)
        else:
            self.save_results(loss_values, self.save_dir, self.runname)
            print(f"Results saved to {self.save_dir}")

        plt.figure()
        plt.plot(
            [loss - self.baseline_loss for loss in loss_values],
            label=self.runname,
            marker='o',
            markevery=int(len(loss_values)/10),
            linestyle="solid",
        )
        plt.xlabel("Global model iteration")
        plt.ylabel(r"$f(x) - f^*$")
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"{self.save_dir}/{self.runname}.png")
        print(f"Plot saved to {self.save_dir}/{self.runname}.png")
        if self.args.verbose:
            plt.show()

        return loss_values

    def save_results(self, loss_values, save_dir, runname):
        # Save the loss values to a CSV file
        with open(f"{save_dir}/{runname}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["global_step", "loss"])
            writer.writerows([[i, loss_values[i]]
                             for i in range(len(loss_values))])

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
    experiment = Experiment(args)
    loss_values = experiment.run_experiment()


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
