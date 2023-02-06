#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train an image classifier with FLSim to simulate a federated learning training environment.

With this tutorial, you will learn the following key components of FLSim:
1. Data loading
2. Model construction
3. Trainer construction

    Typical usage example:
    python3 cifar10_example.py --config-file configs/cifar10_config.json
"""
import random

import tqdm
import flsim.configs  # noqa
import hydra
import torch
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple
from flsim.data.data_provider import IFLDataProvider, IFLUserData

from flsim.data.data_sharder import FLDataSharder, SequentialSharder, PowerLawSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.data.data_utils import batchify
from flsim.utils.example_utils import (
    FLModel,
    MetricsReporter,
    SimpleConvNet,
    UserData,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.vision import VisionDataset

def collate_fn(batch: Tuple) -> Dict[str, Any]:
    feature, label = batch
    return {"features": feature, "labels": label}

IMAGE_SIZE = 32

class DataProvider(IFLDataProvider):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(
            data_loader.fl_eval_set(), eval_split=1.0
        )
        self._test_users = self._create_fl_users(
            data_loader.fl_test_set(), eval_split=1.0
        )

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, iterator: Iterator, eval_split: float = 0.0
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: UserData(user_data, eval_split=eval_split)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }

class DataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: VisionDataset,
        eval_dataset: VisionDataset,
        test_dataset: VisionDataset,
        sharder_train: FLDataSharder,
        sharder_rest: FLDataSharder,
        batch_size: int,
        drop_last: bool = False,
        collate_fn=collate_fn,
    ):
        assert batch_size > 0, "Batch size should be a positive integer."
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sharder_train = sharder_train
        self.sharder_rest = sharder_rest
        self.collate_fn = collate_fn

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        yield from self._batchify(self.train_dataset, self.sharder_train, self.drop_last, world_size, rank)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, self.sharder_rest, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, self.sharder_rest, drop_last=False)

    def _batchify(
        self,
        dataset: VisionDataset,
        sharder: FLDataSharder,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> Generator[Dict[str, Generator], None, None]:
        # pyre-fixme[16]: `VisionDataset` has no attribute `__iter__`.
        data_rows: List[Dict[str, Any]] = [self.collate_fn(batch) for batch in dataset]
        for _, (_, user_data) in enumerate(self.sharder.shard_rows(data_rows)):
            batch = {}
            keys = user_data[0].keys()
            for key in keys:
                attribute = {
                    key: batchify(
                        [row[key] for row in user_data],
                        self.batch_size,
                        drop_last,
                    )
                }
                batch = {**batch, **attribute}
            yield batch


def build_data_provider(local_batch_size, num_users, drop_last: bool = False):

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="../cifar10", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="../cifar10", train=False, download=True, transform=transform
    )
    # sharder = SequentialSharder(examples_per_shard=examples_per_user)
    sharder_train = PowerLawSharder(alpha=0.1, num_shards=num_users)
    sharder_rest = PowerLawSharder(alpha=0.1, num_shards=int(num_users / 5)) # hacky, should be configurable in test
    fl_data_loader = DataLoader(
        train_dataset, test_dataset, test_dataset, sharder_train, sharder_rest, local_batch_size, drop_last
    )
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider


def main(
    trainer_config,
    data_config,
    use_cuda_if_available: bool = True,
) -> None:
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    model = SimpleConvNet(in_channels=3, num_classes=10)
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        num_users=data_config.num_users,
        drop_last=False,
    )

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="cifar10_tutorial")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data

    main(
        trainer_config,
        data_config,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
