"""Downstream expert for Spoken Term Detection on Speech Commands."""

import re
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Union, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from numpy import unique, ndarray
from torch.utils.data import DataLoader, WeightedRandomSampler
from catalyst.data.sampler import DistributedSamplerWrapper
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .dataset import CommonVoiceDataset, CommonVoiceTestingDataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim: int, downstream_expert: dict, expdir: str, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        train_list, valid_list, test_list, CLASSES = split_dataset(self.datarc["common_voice_root"])
        CLASSES = CLASSES.tolist()
        print(CLASSES)
        self.train_dataset = CommonVoiceDataset(train_list, **self.datarc)
        self.dev_dataset = CommonVoiceDataset(valid_list, **self.datarc)
        self.test_dataset = CommonVoiceTestingDataset(test_list, **self.datarc)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=self.train_dataset.class_num,
            **model_conf,
        )

        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    def _get_train_dataloader(self, dataset, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.datarc["batch_size"],
            drop_last=drop_last,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_dev_dataloader(self, dataset, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.datarc["batch_size"],
            drop_last=drop_last,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_test_dataloader(self, dataset):
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.datarc["batch_size"],
            drop_last=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def get_dataloader(self, mode):
        if mode == 'train':
            return self._get_train_dataloader(self.train_dataset, drop_last=True)
        elif mode == 'dev':
            return self._get_dev_dataloader(self.dev_dataset, drop_last=False)
        elif mode == 'test':
            return self._get_test_dataloader(self.test_dataset)
        else:
            raise NotImplementedError

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records["loss"].append(loss.item())
        records["acc"] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        # records["filename"] += filenames
        # records["predict"] += [CLASSES[idx] for idx in predicted_classid.cpu().tolist()]
        # records["truth"] += [CLASSES[idx] for idx in labels.cpu().tolist()]

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["loss", "acc"]:
            values = records[key]
            average = sum(values) / len(values)
            logger.add_scalar(
                f'common_voice/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir, "log.log"), 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        # with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
        #     lines = [f"{f} {i}\n" for f, i in zip(records["filename"], records["predict"])]
        #     file.writelines(lines)
        #
        # with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
        #     lines = [f"{f} {i}\n" for f, i in zip(records["filename"], records["truth"])]
        #     file.writelines(lines)

        return save_names


def split_dataset(root_dir):
    """Split Speech Commands into 3 set.

    Args:
        root_dir: common voice dataset root dir
    Return:
        train_list: [(language_code, audio_path, class_name), ...]
        valid_list: as above
    """
    train_list, valid_list, test_list = [], [], []
    class_index = []
    for entry in Path(root_dir).iterdir():
        train_data_df = pd.read_csv(os.path.join(entry, 'train.tsv'), sep='\t')
        train_data_df.fillna('nan', inplace=True)
        valid_data_df = pd.read_csv(os.path.join(entry, 'dev.tsv'), sep='\t')
        valid_data_df.fillna('nan', inplace=True)
        test_data_df = pd.read_csv(os.path.join(entry, 'test.tsv'), sep='\t')
        test_data_df.fillna('nan', inplace=True)
        class_index.extend(train_data_df.sentence.unique())
        class_index.extend(valid_data_df.sentence.unique())
        class_index.extend(test_data_df.sentence.unique())
        for i in range(len(train_data_df)):
            train_list.append((entry.name, os.path.join(os.path.join(entry, 'clips'), train_data_df.path[i]),
                               train_data_df.sentence[i]))
        for i in range(len(valid_data_df)):
            valid_list.append((entry.name, os.path.join(os.path.join(entry, 'clips'), valid_data_df.path[i]),
                               valid_data_df.sentence[i]))
        for i in range(len(test_data_df)):
            test_list.append((entry.name, os.path.join(os.path.join(entry, 'clips'), test_data_df.path[i]),
                              test_data_df.sentence[i]))
    return train_list, valid_list, test_list, unique(np.asarray(class_index))
