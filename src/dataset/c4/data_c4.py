# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Build a StreamingC4 dataset and dataloader for training.
"""

import os
import sys
from itertools import islice
from typing import Any, Dict, Iterator, Mapping, Optional

import transformers
from omegaconf import OmegaConf as om
from streaming import Dataset
from torch.utils.data import DataLoader


class StreamingC4(Dataset):
    """
    Implementation of the C4 (Colossal Cleaned Common Crawl) dataset using mosaicml-streaming's Dataset V2.
    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        prefetch (int): Target number of samples remaining to prefetch while iterating.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Supports 'truncate' or 'concat'.
        retry (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 prefetch: int,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'truncate',
                 retry: int = 2,
                 timeout: float = 120,
                 batch_size: Optional[int] = None):
        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
        if group_method not in ['truncate', 'concat']:
            raise ValueError(f"group_method='{group_method}' must be one of ['truncate', 'concat'].")

        # Build Dataset
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         prefetch=prefetch,
                         keep_zip=False,
                         retry=retry,
                         timeout=timeout,
                         hash=None,
                         batch_size=batch_size)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        elif self.group_method == 'concat':
            truncation = False
            padding = False
            max_length = None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")
        return self.tokenizer(text_sample['text'], truncation=truncation, padding=padding, max_length=max_length)

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        return token_sample

    # Define iterable over samples
    # Usually this can be left alone and inherited directly from super() class StreamingDataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the token sample.
    # If group_method=='concat', then we keep fetching token samples until we fill up max_seq_len.
    def __iter__(self) -> Iterator[Any]:
        if self.group_method == 'truncate':
            iterator = super().__iter__()
            yield from iterator

        elif self.group_method == 'concat':
            buffer = {}
            while True:
                iterator = super().__iter__()
                for sample in iterator:

                    for k, v in sample.items():
                        buffer[k] = buffer.get(k, []) + v
                    while len(buffer['input_ids']) >= self.max_seq_len:
                        concat_sample = {}
                        for k, v in buffer.items():
                            concat_sample[k] = v[:self.max_seq_len]
                            buffer[k] = v[self.max_seq_len:]
                        yield concat_sample
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")

    # Define length
    # Usually this can be left alone and inherited directly from super() class Dataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the # samples.
    # If group_method=='concat', we repeat forever, and we don't have a defined length.
    def __len__(self) -> int:
        if self.group_method == 'truncate':
            return super().__len__()
        elif self.group_method == 'concat':
            return None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")


def build_c4_dataloader(
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 prefetch: int,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'truncate',
                 retry: int = 2,
                 timeout: float = 120,
                 batch_size: Optional[int] = None,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 num_workers: int = 8,
                 prefetch_factor: int = 2,
                 drop_last: bool = True,
                 ):

    #assert cfg.name == 'c4', f'Tried to build c4 dataloader with cfg.name={cfg.name}'
    dataset = StreamingC4(split=split,
                            remote=remote,
                            local=local,
                            shuffle=shuffle,
                            prefetch=prefetch,
                            tokenizer_name=tokenizer_name,
                            max_seq_len=max_seq_len,
                            group_method=group_method,
                            batch_size=batch_size)

    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer, mlm=False)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        timeout=timeout,
    )