#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train a next-char predictor on LEAF's Shakespeare dataset with FLSim.

Before running this file, you need to download the dataset, and partition the data by users. We
provided the script get_data.sh for such task.

    Typical usage example:

    FedAvg
    CUDA_VISIBLE_DEVICES=2, python3 shakespeare.py --config-file configs/shakespeare.json
"""
import os
import json
import re

import flsim.configs  # noqa
import hydra  # @manual
import torch
import torch.nn as nn
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


class LSTMModel(nn.Module):
    def __init__(
        self,
        num_classes,
        n_hidden=256,
        num_embeddings=80,
        embedding_dim=8,
        seq_len=80,
        dropout_rate=0,
    ):
        super(LSTMModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        return self.fc(self.dropout(final_hidden_state))


class ShakespeareDataset(Dataset):
    def __init__(self, data_root, seq_len=80):
        self.ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.NUM_LETTERS = len(self.ALL_LETTERS)
        self.data_root = data_root
        self.seq_len = seq_len

        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        self.data = {}
        self.targets = {}
        self.num_classes = 80  # next char prediction

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
        x_batch = [self.word_to_indices(word) for word in raw_x_batch]
        return torch.LongTensor(x_batch)

    def process_y(self, raw_y_batch):
        y_batch = [self.letter_to_index(c) for c in raw_y_batch]
        return torch.LongTensor(y_batch)

    # ------------------------
    # utils for shakespeare dataset

    def _one_hot(self, index, size):
        '''returns one-hot vector with given size and value 1 at given index
        '''
        vec = [0 for _ in range(size)]
        vec[int(index)] = 1
        return vec

    def letter_to_index(self, letter):
        '''returns index of given letter in ALL_LETTERS
        '''
        return self.ALL_LETTERS.find(letter)

    def letter_to_vec(self, letter):
        '''returns one-hot representation of given letter
        '''
        index = self.ALL_LETTERS.find(letter)
        return self._one_hot(index, self.NUM_LETTERS)

    def word_to_indices(self, word):
        '''returns a list of character indices

        Args:
            word: string

        Return:
            indices: int list with length len(word)
        '''
        indices = []
        for c in word:
            indices.append(self.ALL_LETTERS.find(c))
        return indices


def build_data_provider(data_config, drop_last=False):

    MAIN_DIR = './../../leaf/data/shakespeare'
    train_dir = os.path.join(MAIN_DIR, 'data/train')
    TRAIN_DATA = os.path.join(train_dir, os.listdir(train_dir)[0])

    test_dir = os.path.join(MAIN_DIR, 'data/test')
    TEST_DATA = os.path.join(test_dir, os.listdir(test_dir)[0])

    train_dataset = ShakespeareDataset(
        data_root=TRAIN_DATA,
        seq_len=data_config.seq_len,
    )
    test_dataset = ShakespeareDataset(
        data_root=TEST_DATA,
        seq_len=data_config.seq_len,
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=drop_last,
    )

    data_provider = DataProvider(dataloader)
    return data_provider


def main_worker(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
):
    data_provider = build_data_provider(data_config)

    model = LSTMModel(
        num_classes=model_config.num_classes,
        n_hidden=model_config.n_hidden,
        num_embeddings=80,
        embedding_dim=8,
        seq_len=80,
        dropout_rate=model_config.dropout_rate,
    )

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device("cuda" if cuda_enabled else "cpu")
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(
        trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=distributed_world_size,
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter(
            [Channel.TENSORBOARD, Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="shakespeare_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    main_worker(
        cfg.trainer,
        cfg.model,
        cfg.data,
        use_cuda_if_available=cfg.use_cuda_if_available, distributed_world_size=cfg.distributed_world_size
    )


def main() -> None:
    cfg = maybe_parse_json_config()
    run(cfg)


if __name__ == "__main__":
    main()  # pragma: no cover