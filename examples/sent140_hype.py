#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""In this tutorial, we will train a binary sentiment classifier on LEAF's Sent140 dataset with FLSim.

Before running this file, you need to download the dataset, and partition the data by users. We
provided the script get_data.sh for such task.

    Typical usage example:

    FedAvg
    python3 sent140_tutorial.py --config-file configs/sent140_config.json

    FedBuff + SGDM
    python3 sent140_tutorial.py --config-file configs/sent140_fedbuff_config.json
"""
import json
import re
import os
import random

import flsim.configs  # noqa
import hydra  # @manual
import torch
import torch.nn as nn
import numpy as np
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataProvider,
    FLModel,
    MetricsReporter,
    LEAFDataLoader,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from skopt import gp_minimize
from skopt.space import Real

from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=300, max_vectors=10000)

def get_word_emb(path):
    return glove.stoi, len(glove.stoi)

class LSTMModel(nn.Module):
    def __init__(
        self, seq_len, num_classes, embedding_dim, n_hidden, dropout_rate, vocab_size
    ):
        super(LSTMModel, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        glove_vectors = glove.vectors
        embedding_matrix = torch.zeros((vocab_size + 1, embedding_dim))
        embedding_matrix[:vocab_size] = glove_vectors
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
        # self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim)
        self.stacked_lstm = nn.LSTM(
            self.embedding_dim,
            self.n_hidden,
            2,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.fc1 = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.out = nn.Linear(128, self.num_classes)

    """
    def forward(self, features):
        seq_lens = torch.sum(features != (self.vocab_size - 1), 1) - 1
        x = self.embedding(features)
        outputs, _ = self.stacked_lstm(x)
        outputs = outputs[torch.arange(outputs.size(0)), seq_lens]
        pred = self.fc1(self.dropout(outputs))
        return pred
    """ 
    def forward2(self, features):
        seq_lens = torch.sum(features != (self.vocab_size), 1) - 1
        x = self.embedding(features)
        outputs, _ = self.stacked_lstm(x)
        outputs = outputs[torch.arange(outputs.size(0)), seq_lens]
        pred = self.fc1(self.dropout(outputs))
        return pred
    
    def forward(self, features):
        x = self.embedding(features)
        x, _ = self.stacked_lstm(x)
        x = self.fc1(self.dropout(x[:, -1]))
        return x

class Sent140Dataset(Dataset):
    def __init__(self, data_root, max_seq_len, vocab_dir):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        self.data = {}
        self.targets = {}

        self.num_classes = 2
        self.word2id, self.vocab_size = get_word_emb(vocab_dir)

        # Populate self.data and self.targets
        for user_id, user_data in self.dataset["user_data"].items():
            self.data[user_id] = self.process_x(list(user_data["x"]))
            self.targets[user_id] = self.process_y(list(user_data["y"]))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str):
        if user_id not in self.data or user_id not in self.targets:
            raise IndexError(f"User {user_id} is not in dataset")

        return self.data[user_id], self.targets[user_id]

    def process_x(self, raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [self.line_to_indices(e, self.max_seq_len) for e in x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        return y_batch

    def line_to_indices(self, line, max_words=25):
        """Converts given phrase into list of word indices
        
        if the phrase has more than max_words words, returns a list containing
        indices of the first max_words words
        if the phrase has less than max_words words, repeatedly appends integer 
        representing unknown index to returned list until the list's length is 
        max_words
        """
        unk_id = len(self.word2id)
        line_list = self.split_line(line)  # split phrase in words
        indl = [
            self.word2id[w] if w in self.word2id else unk_id
            for w in line_list[:max_words]
        ]
        indl += [unk_id]*(max_words-len(indl))
        return indl

    def split_line(self, line):
        return re.findall(r"[\w']+|[.,!?;]", line)


def build_data_provider(data_config, drop_last=False):

    data_root = os.path.join(data_config.leaf_dir, "data/sent140")

    train_dataset = Sent140Dataset(
        data_root=os.path.join(
            data_root, "data/train/all_data_0_15_keep_10_train_9.json"),
        max_seq_len=data_config.max_seq_len,
        vocab_dir=data_config.vocab_dir,
    )
    test_dataset = Sent140Dataset(
        data_root=os.path.join(
            data_root, "data/test/all_data_0_15_keep_10_test_9.json"),
        max_seq_len=data_config.max_seq_len,
        vocab_dir=data_config.vocab_dir,
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=drop_last,
    )

    data_provider = DataProvider(dataloader)
    return data_provider, train_dataset.vocab_size


def main_worker(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
    log_dir=None,
    data_provider=None,
    vocab_size=None,
):
    # Print the configuration for debugging
    print(OmegaConf.to_yaml(trainer_config))
    print(OmegaConf.to_yaml(model_config))

    model = LSTMModel(
        num_classes=model_config.num_classes,
        n_hidden=model_config.n_hidden,
        vocab_size=vocab_size,
        embedding_dim=300,
        seq_len=data_config.max_seq_len,
        dropout_rate=model_config.dropout_rate,
    )

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device("cuda" if cuda_enabled else "cpu")
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(
        trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    metrics_reporter = MetricsReporter(
        [Channel.TENSORBOARD, Channel.STDOUT], log_dir=log_dir)

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=distributed_world_size,
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter(
            [Channel.TENSORBOARD, Channel.STDOUT], log_dir=log_dir),
    )
    
    print(eval_score)
    return -eval_score['Accuracy']

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


@hydra.main(config_path=None, config_name="sent140_tutorial")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seeds(cfg.seed)

    data_provider, vocab_size = build_data_provider(cfg.data)

    # Define the search space for hyperparameters
    space = [
        Real(1e-2, 1e2, prior='log-uniform', name='eta_l'),
        Real(1e-2, 1e2, prior='log-uniform', name='eta_g'),
    ]

    def objective(params):
        eta_l, eta_g = params
        cfg.trainer.client.optimizer.lr = eta_l
        cfg.trainer.aggregator.lr = eta_g
        return main_worker(
            cfg.trainer,
            cfg.model,
            cfg.data,
            use_cuda_if_available=cfg.use_cuda_if_available, 
            distributed_world_size=cfg.distributed_world_size,
            log_dir=cfg.log_dir,
            data_provider=data_provider,
            vocab_size=vocab_size,
        )

    # Perform Bayesian optimization
    res = gp_minimize(objective, space, n_calls=100, random_state=0, verbose=10)

    print("Best hyperparameters found: ")
    print("Learning Rate:", res.x[0])
    print("Global Learning Rate:", res.x[1])


def main() -> None:
    cfg = maybe_parse_json_config()
    run(cfg)


if __name__ == "__main__":
    main()  # pragma: no cover
