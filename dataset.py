"""TODO:
* remove own embedding and use nn.Embedding

* Follow this source for handling variable length data. Source:
  https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418
  It will be more memory efficient.
"""

from typing import List, Tuple, Optional, Dict, Set
import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data as tud
import random

import util.collect_dataset as col_dat

PAD_TOKEN_VALUE = len(col_dat.TOKEN_FREQUENCIES)

# from dataclasses import dataclass

# TODO(aleloi): consider using torch WeightedRandomSampler to draw
# samples from each class equally often.
class Dataset(tud.Dataset):
    def __init__(self, /,
                 validation_proportion : float,
                 flatten : bool,
                 #reweight : bool, - a data loader or sampler thing.
                 is_training_not_validation: bool,
                 take_top10: bool,
                 drop_large_threshold_tokens: Optional[int] = None,
                 random_seed: int = 0,
                 max_per_class: Optional[int] = None
                 ):
        """Remaps tokens to 1-hot encoded numbers (so that we can decide later
        whether to project them to something low-dim or not).

        Remaps targets to 1-hot encoded classes as well.

        If 'is_training_not_validation' returns training set,
        otherwise validation set. The seed is used to make the
        training and validation set disjoint. Call with same seed:

          train = Dataset(*args, random_seed=seed, is_training_not_validation=True)
          val   = Dataset(*args, random_seed=seed, is_training_not_validation=False)

        """
        # TODO(aleloi): Possibly add 'drop tokens' option? To skip
        # some of the tokens.

        random.seed(random_seed)
        
        self.problems = (col_dat.MOST_COMMON10
                         if take_top10
                         else col_dat.MOST_COMMON)
        self.token_to_int = {tok: i for (i, tok) in
                             enumerate(col_dat.TOKEN_FREQUENCIES.keys())}
        
        NUM_TOKENS = len(self.token_to_int)
        
        assert flatten # and take_top10

        self.programs : List[torch.Tensor] = []
        classes: List[int] = []

        # For test: contains indices of selected solutions to check that
        # val/test indices do not overlap.
        self.indices : Set[Tuple[int, int]] = set()

        accept_in_set = lambda p: p < validation_proportion if is_training_not_validation else p >= validation_proportion

        for problem_num, (contest, letter) in enumerate(self.problems):
            filepath = os.path.join('data', f'cont{contest}_prob{letter}.txt')
            num_programs = 0
            for problem_idx, line in enumerate(
                    open(filepath, 'r').readlines()):
                if max_per_class is not None and num_programs >= max_per_class:
                    break
                if flatten:
                    for char in "()[]'":
                        line = line.replace(char, '')
                    tokens = line.strip().split(', ')
                    if drop_large_threshold_tokens is not None and len(tokens) > drop_large_threshold_tokens:
                        continue
                    num_programs += 1
                    # Do not encode other than to 0...|vocab|, embedding is
                    # done by the Model. Do not pad, that's done by
                    # the DataLoader
                                        
                    token_nums = torch.tensor(
                        [self.token_to_int[tok] for tok in tokens])
                    if accept_in_set(random.random()):
                        self.programs.append(token_nums)
                        classes.append(problem_num)
                        self.indices.add((problem_idx, problem_num))
                else: assert False
        self.class_tensor : torch.Tensor = torch.tensor(
            classes, dtype=torch.long)
        
    def __getitem__(self, key) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (unpadded program, label as a scalar value)
        """
        return (self.programs[key], self.class_tensor[key])

    def __len__(self):
        return len(self.programs)


    
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads programs of inhomogeneous length to the maximum length"""
    progs = [prog for prog, _ in batch]
    lengths = [len(prog) for prog in progs]
    max_len = max(lengths)
    progs_padded = torch.nn.utils.rnn.pad_sequence(
        progs, padding_value=PAD_TOKEN_VALUE)
    assert progs_padded.size() == (max_len, len(batch))
    labels = torch.stack([label for _, label in batch])
    return progs_padded, torch.tensor(lengths), labels


def create_data_loader(ds: Dataset, BATCH_SIZE: int) -> tud.DataLoader:
    return tud.DataLoader(ds, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, shuffle=True)


if __name__ == '__main__':
    # Simple tests:
    from collections import Counter

    MAX_PER_CLASS=10

    args = {'validation_proportion': 0.8, 'flatten': True,
            'take_top10': False,
            'drop_large_threshold_tokens': 200, 'max_per_class': MAX_PER_CLASS}

    train = Dataset(is_training_not_validation = True,
                    **args)  # type: ignore
    val = Dataset(is_training_not_validation = False,
                  **args)  # type: ignore

    C_train : Dict[int, int] = Counter()
    C_val: Dict[int, int] = Counter()
    for prog, label in train:  # type: ignore
        C_train[int(label)] += 1
    for prog, label in val:  # type: ignore
        C_val[int(label)] += 1

    for x in set(C_train.keys()) | set(C_val.keys()):
        assert C_train[x] + C_val[x] == 10, (x, C_train[x] + C_val[x])

    assert train.indices & val.indices == set()
    print(train.indices)
    print(val.indices)
    
    BATCH_SIZE = 2
    dl_train = create_data_loader(train, BATCH_SIZE)
    dl_val = create_data_loader(val, BATCH_SIZE)

    def drop_last(generator):
        cand = next(generator, None)
        while True:
            nxt_cand = next(generator, None)
            if nxt_cand is None: break
            yield cand
            cand = nxt_cand

    for progs,lens,labels in drop_last(x for x in dl_train):
        assert len(labels) == BATCH_SIZE, len(labels)
    for progs,lens,labels in drop_last(x for x in dl_val):
        assert len(labels) == BATCH_SIZE, len(labels)
    
