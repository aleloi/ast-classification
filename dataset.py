"""TODO:
* remove own embedding and use nn.Embedding

* Follow this source for handling variable length data. Source:
  https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418
  It will be more memory efficient.
"""

from typing import List, Tuple
import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data as tud

import util.collect_dataset as col_dat

PAD_TOKEN_VALUE = len(col_dat.TOKEN_FREQUENCIES)

# from dataclasses import dataclass

# TODO(aleloi): consider using torch WeightedRandomSampler to draw
# samples from each class equally often.
class Dataset(tud.Dataset):
    def __init__(self,
                 validation_proportion : float,
                 flatten : bool,
                 #reweight : bool, - a data loader or sampler thing.
                 take_top10: bool,
                 drop_large_threshold_tokens: int
                 ):
        """Remaps tokens to 1-hot encoded numbers (so that we can decide later
        whether to project them to something low-dim or not).

        Remaps targets to 1-hot encoded classes as well.

        """
        # TODO(aleloi): Possibly add 'drop tokens' option? To skip
        # some of the tokens.
        
        self.problems = (col_dat.MOST_COMMON10
                         if take_top10
                         else col_dat.MOST_COMMON)
        self.token_to_int = {tok: i for (i, tok) in
                             enumerate(col_dat.TOKEN_FREQUENCIES.keys())}
        
        NUM_TOKENS = len(self.token_to_int)
        
        assert flatten and take_top10

        self.programs : List[torch.Tensor] = []
        classes: List[int] = []

        for problem_num, (contest, letter) in enumerate(self.problems):
            filepath = os.path.join('data', f'cont{contest}_prob{letter}.txt')
            num_programs = 0
            for line in open(filepath, 'r').readlines():
                if num_programs > 1000: break
                num_programs += 1
                if flatten:
                    for char in "()[]'":
                        line = line.replace(char, '')
                    tokens = line.strip().split(', ')
                    if len(tokens) > drop_large_threshold_tokens:
                        continue
                    # Do not encode other than to 0...|vocab|, embedding is
                    # done by the Model. Do not pad, that's done by
                    # the DataLoader
                                        
                    token_nums = torch.tensor(
                        [self.token_to_int[tok] for tok in tokens])
                    self.programs.append(token_nums)
                    classes.append(problem_num)
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

# Example:
#d = Dataset(0, True,  True, 1000)
#dl = create_data_loader(d, 2)
