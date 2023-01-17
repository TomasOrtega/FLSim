#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train a binary sentiment classifier on LEAF's Sent140 dataset with FLSim.

Before running this file, you need to download the dataset and partition the data by users. We
provide the script get_data.sh for this purpose.

    Typical usage example:

    FedAvg
    python3 sent140_example.py --config-file configs/sent140_config.json

    FedBuff + SGDM
    python3 sent140_example.py --config-file configs/sent140_fedbuff_config.json

    FedBuff + SGDM + quantization
    python3 sent140_example.py --config-file configs/sent140_quantized_fedbuff_config.json
"""
import itertools
import json
import re
import string
import unicodedata
from typing import List

import flsim.configs  # noqa
import hydra  # @manual
import torch
import torch.nn as nn
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataProvider,
    FLModel,
    LEAFDataLoader,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer


class CharLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        n_hidden,
        num_embeddings,
        embedding_dim,
        max_seq_len,
        dropout_rate,
        glove,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.num_classes = num_classes
        
        embedding_vectors = torch.zeros([num_embeddings + 1, embedding_dim], dtype=torch.float32)
        print(glove.vectors.shape, embedding_vectors.shape)
        embedding_vectors[:-1, :] = glove.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_vectors)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        # self.fc = nn.Linear(self.n_hidden, 128)
        # self.dropout1 = nn.Dropout(p=self.dropout_rate)
        # self.pred = nn.Linear(128, self.num_classes)
        # self.dropout2 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = self.embedding(x)  # [B, S] -> [B, S, E]
        out, _ = self.lstm(x)  # [B, S, E] -> [B, S, H]
        out = out[:, -1, :] # slice lstm_out to just get output of last element of the input sequence
        out = self.fc(out)
        # out = self.fc(self.dropout1(out))  # [B, S, H] -> # [B, S, C]
        # out = self.pred(self.dropout2(out))  # [B, S, H] -> # [B, S, C]
        return out


class Sent140Dataset(Dataset):
    def __init__(self, data_root, max_seq_len, glove, num_embeddings):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer("basic_english")
        self.stoi = glove.stoi # string to index dictionary
        self.num_embeddings = num_embeddings

        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        self.data = {}
        self.targets = {}

        self.num_classes = 2

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

    def line_to_tokens(self, line: str, max_seq_len: int):
        tokens = self.tokenizer(line)  # split phrase in tokens
        # padding
        if len(tokens) >= max_seq_len:
            tokens = tokens[:max_seq_len]
        else:
            tokens = tokens + [""] * (max_seq_len - len(tokens))
        return tokens
    
    def token_to_index(self, token):
        if token in self.stoi:
            return self.stoi[token]
        else:
            return self.num_embeddings

    def tokens_to_indices(self, tokens):
        indices = [self.token_to_index(token) for token in tokens]
        return indices

    def process_x(self, raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [self.line_to_tokens(e, self.max_seq_len) for e in x_batch]
        x_batch = [self.tokens_to_indices(e) for e in x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        return y_batch


def build_data_provider(data_config, glove, drop_last: bool = False, num_embeddings: int = 10000):
    
    train_dataset = Sent140Dataset(
        data_root="../../leaf/data/sent140/data/train/all_data_0_15_keep_1_train_8.json",
        max_seq_len=data_config.max_seq_len,
        glove=glove,
        num_embeddings=num_embeddings
    )
    test_dataset = Sent140Dataset(
        data_root="../../leaf/data/sent140/data/test/all_data_0_15_keep_1_test_8.json",
        max_seq_len=data_config.max_seq_len,
        glove=glove,
        num_embeddings=num_embeddings
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
    use_cuda_if_available: bool = False,
    distributed_world_size: int = 1,
) -> None:
    # Glove pre-trained embedding
    glove_dim = 300
    num_embeddings = 10000
    glove = GloVe(name="6B", dim=glove_dim, max_vectors=num_embeddings) # as per FedBuff paper
    data_provider = build_data_provider(data_config, glove, drop_last=False, num_embeddings=num_embeddings)
    model = CharLSTM(
        num_classes=model_config.num_classes,
        n_hidden=model_config.n_hidden,
        num_embeddings=num_embeddings,
        embedding_dim=glove_dim,
        max_seq_len=data_config.max_seq_len,
        dropout_rate=model_config.dropout_rate,
        glove=glove
    )
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=distributed_world_size,
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="sent140_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    main_worker(trainer_config, model_config, data_config)


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
