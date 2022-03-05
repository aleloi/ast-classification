"""TODO:
* remove own embedding and use nn.Embedding

* Follow this source for handling variable length data. Source:
  https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418
  It will be more memory efficient.
"""

from typing import List, Tuple, Optional, Dict, Set, Union, Any
import typing
import os
import torch
import torch.utils.data as tud
import ast
import dgl  # type: ignore

import util.collect_dataset as col_dat


PAD_TOKEN_VALUE = len(col_dat.TOKEN_FREQUENCIES)

TOKEN_TO_INT = {tok: i for (i, tok) in
                enumerate(
                    sorted(col_dat.TOKEN_FREQUENCIES.keys()))}

StrTree = Tuple[str, list]
StrTreeOrLeaf = Union[Tuple[str, list], str]

def to_dgl_tree(line: str) -> dgl.DGLGraph:
    tree: StrTree = ast.literal_eval(line)
    from_l: List[int]
    to_l: List[int]
    tok_l: List[int]
    from_l, to_l, tok_l = [], [], []
    compute_edges_rec(tree, from_l, to_l, tok_l, None)
    g = dgl.graph((torch.tensor(from_l), torch.tensor(to_l)))
    g.ndata['x'] = torch.tensor(tok_l)
    return g

def compute_edges_rec(prog: StrTreeOrLeaf,
                      from_l: list, to_l: list, tok_l: list,
                      parent_idx=None):
    """Computes an edge-list representation of the tree."""
    children: List[StrTreeOrLeaf]
    if isinstance(prog, str):
        token, children = prog, []
    else:
        token, children = prog
    idx = len(tok_l)
    tok_l.append(TOKEN_TO_INT[token])
    if parent_idx is not None:
        from_l.append(idx)
        to_l.append(parent_idx)

    for child in children:
        compute_edges_rec(child, from_l, to_l, tok_l, idx)

def _flatten_program_to_token_list(line: str) -> List[str]:
    for char in "()[]'":
        line = line.replace(char, '')
    return line.strip().split(', ')

def flatten_program(line: str) -> torch.Tensor:
    tokens = _flatten_program_to_token_list(line)
    return  torch.tensor(
        [TOKEN_TO_INT[tok] for tok in tokens])
        

# TODO(aleloi): consider using torch WeightedRandomSampler to draw
# samples from each class equally often.
class Dataset(tud.Dataset):
    def __init__(self, /,
                 flatten : bool,
                 take_top10: bool,
                 do_prune_duplicates: bool,
                 drop_large_threshold_tokens: Optional[int] = None,
                 max_per_class: Optional[int] = None
                 ):
        """Converts the ASTs in 'data/' to DGL-trees with tokens mapped to
        consecuite integers. Optionally (when `flatten=True`) produces
        lists of token ids instead.

        Filters 

        """
        # TODO(aleloi): Possibly add 'drop tokens' option? To skip
        # some of the tokens.
        self.max_per_class = max_per_class
        self.problems = sorted(col_dat.MOST_COMMON10
                               if take_top10
                               else col_dat.MOST_COMMON)
        self.num_classes = len(self.problems)
        self.has_program_trees = not flatten

        self.programs : List[torch.Tensor] = []
        self.program_trees : List[dgl.DGLGraph] = []
        classes: List[int] = []
        num_large_dropped = 0

        prune_uniq = '.uniq.txt' if  do_prune_duplicates else ''
        
        for problem_num, (contest, letter) in enumerate(self.problems):
            filepath = os.path.join('data', f'cont{contest}_prob{letter}.txt{prune_uniq}')
            num_programs = 0
            for problem_idx, line in enumerate(
                    open(filepath, 'r').readlines()):
                if max_per_class is not None and num_programs >= max_per_class:
                    break
                num_tokens = len(_flatten_program_to_token_list(line))
                if (drop_large_threshold_tokens is not None and
                    num_tokens > drop_large_threshold_tokens):
                    num_large_dropped += 1
                    continue
                
                if flatten:
                    self.programs.append(flatten_program(line))
                else:
                    self.program_trees.append(to_dgl_tree(line))
                num_programs += 1

                # Special case that's messy to fix: turns out problem
                # 11 and 31 are the same promle,
                # https://codeforces.com/contest/1465/problem/A and
                # https://codeforces.com/contest/1411/problem/A. I
                # have a bunch of pre-trained models and rely on the
                # exact class numbers, so adding a different problem
                # would mess up the classes. Therefore, just relabel
                # them, and use 103 classes instead of 104.
                if problem_num == 31:
                    classes.append(11)
                else:
                    classes.append(problem_num)

                if (len(self.programs) + len(self.program_trees)) % 10000 == 0:
                    print(f"In dataset; parsing problem {problem_idx} "
                          f"from {contest}{letter} "
                          f"[{problem_num}/{len(self.problems)}]; have "
                          f"{len(self.programs)+len(self.program_trees)} "
                          "programs.\n"
                          f"Total large dropped: {num_large_dropped}"
                          )

            if self.max_per_class is not None:
                assert num_programs == self.max_per_class, num_programs
                        
        self.class_tensor : torch.Tensor = torch.tensor(
            classes, dtype=torch.long)

    def compute_class_weights(self, class_labels):
        """If max_per_class is None, we go though the complete dataset, then
        draw samples from a weighted distribution, so that drawing
        each class is equally likely.
        """
        #assert self.max_per_class is None
        count = torch.bincount(self.class_tensor)
        # count is {n_i}_i - amount of values per class.
        assert count.size() == (self.num_classes,), (
            count.size(), self.num_classes)
        class_weights = 1/count

        # The index 11 vs 31 bug (see comment above).
        if self.num_classes > 10:
            class_weights[31] = 0.
        print(f"Class weights are: {class_weights}")
        weights = torch.tensor([class_weights[problem_class]
                                for problem_class in class_labels])
        return weights
        
        
    def __getitem__(self, key) -> Tuple[
            Union[torch.Tensor, dgl.DGLGraph],
            torch.Tensor]:
        """Returns (unpadded program, label as a scalar value)
        """
        if self.has_program_trees:
            return (self.program_trees[key], self.class_tensor[key])
        else: 
            return (self.programs[key], self.class_tensor[key])

    def __len__(self):
        if self.has_program_trees:
            return len(self.program_trees)
        else:
            return len(self.programs)

def flattened_collate_fn(batch: List[Tuple[torch.Tensor,
                                           torch.Tensor]]) -> Tuple[
                                               torch.Tensor,
                                               torch.Tensor,
                                               torch.Tensor]:
    """Pads programs of inhomogeneous length to the maximum length"""
    progs = [prog for prog, _ in batch]
    lengths = [len(prog) for prog in progs]
    max_len = max(lengths)
    progs_padded = torch.nn.utils.rnn.pad_sequence(
        progs, padding_value=PAD_TOKEN_VALUE)
    assert progs_padded.size() == (max_len, len(batch))
    labels = torch.stack([label for _, label in batch])
    return progs_padded, torch.tensor(lengths), labels

class DataArgs(typing.NamedTuple):
    flatten: bool
    take_top10: bool
    training_weight : float
    validation_weight: float
    test_weight: float
    train_samples: Optional[int]
    val_samples: Optional[int]
    test_samples: Optional[int]
    small_train_samples: int
    batch_size: int
    do_prune_duplicates: bool = True
    drop_large_threshold_tokens: Optional[int] = None
    max_per_class: Optional[int] = None
    

def get_datasets(opts: DataArgs) -> Tuple[tud.DataLoader,
                                          tud.DataLoader,
                                          tud.DataLoader,
                                          tud.DataLoader] :
    """
    Returns train, small train (for gradients), val and test
    """
    full_ds = Dataset(
        flatten=opts.flatten, take_top10=opts.take_top10,
        drop_large_threshold_tokens=opts.drop_large_threshold_tokens,
        max_per_class=opts.max_per_class,
        do_prune_duplicates=opts.do_prune_duplicates
    )
    tot_w  = opts.training_weight + opts.validation_weight + opts.test_weight
    training_weight, validation_weight, test_weight = (
        opts.training_weight / tot_w,
        opts.validation_weight / tot_w,
        opts.test_weight / tot_w)
    len_train = int(len(full_ds)*training_weight)
    len_val = int(len(full_ds)*validation_weight)
    len_test = int(len(full_ds) - len_train - len_val)
    d_train, d_val, d_test = torch.utils.data.random_split(
        full_ds,
        [len_train, len_val, len_test],
        generator=torch.Generator().manual_seed(0)
    )

    dls: List[tud.DataLoader] = []
    dl_args : Dict[str, Any] = {
        'batch_size': opts.batch_size,
        'collate_fn': (flattened_collate_fn
                       if opts.flatten
                       else (lambda x: x))  # type: ignore
    }
    for ds, samples_per_epoch in zip(
            [d_train, d_val, d_test],
            [opts.train_samples, opts.val_samples, opts.test_samples]):
                           
        if samples_per_epoch is None:
            dls.append(tud.DataLoader(ds, **dl_args))
        else:
            train_labels = torch.stack(
                [lbl for _, lbl in ds])  # type: ignore
            sampler=tud.WeightedRandomSampler(
                full_ds.compute_class_weights(train_labels),
                samples_per_epoch,
                generator=torch.Generator().manual_seed(0))
            dls.append(tud.DataLoader(ds, sampler=sampler, **dl_args))
    dl_train, dl_val, dl_test = dls

    small_train_labels = torch.stack(
                [lbl for _, lbl in d_train])  # type: ignore

    sampler=tud.WeightedRandomSampler(
        full_ds.compute_class_weights(small_train_labels),
        opts.small_train_samples,
        generator=torch.Generator().manual_seed(0)
        )
    dl_train_small = tud.DataLoader(d_train,
                                    sampler=sampler,
                                    **dl_args)
    
    print(f"Train Dataset length: {len(d_train)}")
    tot_num_batches = len(dl_train)
    print(f"Train Dataloader length (num batches): {tot_num_batches}")
    print(f"Val Dataset length: {len(d_val)}")
    print(f"Val Dataloader length: {len(dl_val)}")
    print(f"Small dataloader has {len(dl_train_small)} batches")
    
    return (dl_train, dl_train_small, dl_val, dl_test)
    

if __name__ == '__main__':
    # Simple tests:
    from collections import Counter

    BATCH_SIZE = 20
    
    da = DataArgs(flatten=True, take_top10=True,
                  training_weight = 3,
                  validation_weight = 1,
                  test_weight = 1,
                  train_samples = 100,
                  val_samples = 20,
                  test_samples = 20,
                  batch_size=BATCH_SIZE,
                  small_train_samples=1,
                  max_per_class=10)

    dl_train, dl_small_train, dl_val, dl_test = get_datasets(da)

    gen = (x for x in dl_train)
    print(next(gen))
        
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

    # Tree test:
    da = DataArgs(flatten=False, take_top10=True,
                  training_weight = 3,
                  validation_weight = 1,
                  test_weight = 1,
                  train_samples = 100,
                  val_samples = 20,
                  test_samples = 20,
                  small_train_samples=1,
                  batch_size=16,
                  max_per_class=10)
    
    dl_train, _, dl_val, dl_test = get_datasets(da)
        
    for bt in dl_train:
        print(bt)
        break
