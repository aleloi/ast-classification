import dgl_lstm_model as dgl_model
import linear_lstm_model as linear_model
import dataset
import utils

import pathlib
import dgl  # type: ignore
import sys
import torch.nn as nn
import torch
import torch.utils.data as tud
from typing import Union, Tuple, Optional, Dict, Any
import datetime
from tensorboardX import SummaryWriter  # type: ignore


def train_epoch(model, #: tree_model.DGLTreeLSTM,
                optimizer: torch.optim.Optimizer,
                train_dl: tud.DataLoader
                ):
    tot_num_batches = len(train_dl)
    B = utils.batch_size(train_dl).to(model.device)
    
    sum_loss = torch.tensor(0., requires_grad=False).to(model.device)
    num_correct = torch.tensor(0, requires_grad=False).to(model.device)

    training_start = datetime.datetime.now()
    tot_num_samples = 0
    is_linear = isinstance(model, linear_model.LinearLSTM)
    for batch_idx, batch in enumerate(train_dl):
        batch_training_start = datetime.datetime.now()
        optimizer.zero_grad()
        if is_linear:
            programs, lengths, targets = batch
            targets = targets.to(model.device)
            logits_t = model(programs, lengths)
            #breakpoint()
        else:
            progs = dgl.batch([prog for prog, _ in batch]).to(model.device)
            targets = torch.stack([cls for _, cls in batch]).to(
                model.device)
            logits_t = model(progs) 
        loss = model.train_loss(logits_t, targets)
        if is_linear:
            metrics = utils.compute_batch_metrics_linear(
                model, logits_t, loss,
                (programs, lengths, targets))
        else:
            metrics = utils.compute_batch_metrics_tree(
                model, logits_t, loss,
                batch)
        num_correct += metrics['num_correct']
        sum_loss += metrics['loss'] * B
        tot_num_samples += len(batch)
        loss.backward()
        optimizer.step()
        if ((batch_idx % 300 == 0 and batch_idx > 0)
            or (batch_idx % 30 == 0 and batch_idx < 300)):
            print(f"Statistics for batch {batch_idx+1} of "
                  f"{tot_num_batches}:")
            samples_this_batch = float(B * (batch_idx+1))
            print(f"\t Average loss this epoch: "
                  f"{float(sum_loss / (batch_idx+1))}")
            print(f"\t Average accuracy this epoch: "
                  f"{float(num_correct / samples_this_batch)}")
            now = datetime.datetime.now()
            Δ = now - training_start
            print(f"\t Time per sample (this batch): "
                  f"{Δ / samples_this_batch}")
            print(f"\t Samples per second (this batch): "
                  f"{samples_this_batch / max(Δ.seconds, 1)}")
        
    num_batches = torch.tensor(float(batch_idx)).to(model.device)
    avg_loss = sum_loss / num_batches
    avg_accuracy = num_correct / (num_batches * B)
    return {'avg_loss': avg_loss, 'avg_accuracy': avg_accuracy}
    
def train(model, epoch: int,
          num_epochs : int,
          optimizer: torch.optim.Optimizer,
          dl_train: tud.DataLoader ,
          dl_train_small: tud.DataLoader,
          dl_val: tud.DataLoader,
          results_dir: str):
    i = epoch+1
    is_linear = isinstance(model, linear_model.LinearLSTM)
    writer = SummaryWriter(results_dir)
    while i < num_epochs:
        print(f"\nEPOCH: {i+1}/{num_epochs}:")
        metrics_train = train_epoch(model, optimizer, dl_train)
        writer.add_scalar('train/avg_loss',
                          metrics_train['avg_loss'], i)
        writer.add_scalar('train/avg_accuracy',
                          metrics_train['avg_accuracy'], i)
        if is_linear:
            metrics_val = utils.validation_metrics_linear(model, dl_val)
        else:
            metrics_val = utils.validation_metrics_tree(model, dl_val)
        writer.add_scalar('val/avg_loss',
                          metrics_val['avg_loss'], i)
        writer.add_scalar('val/avg_accuracy',
                          metrics_val['avg_accuracy'], i)
    
        print(f"train loss: {metrics_train['avg_loss']:.3f}")
        print(f"train accuracy: {metrics_train['avg_accuracy']*100:.2f}")
        print(f"val loss: {metrics_val['avg_loss']:.3f}")
        print(f"val accuracy: {metrics_val['avg_accuracy']*100:.2f}")
        utils.compute_embeddings(
            writer, model,
            model_name=pathlib.Path(results_dir).parent.parts[-1],
            epoch=i,
            do_load=False
        )
        utils.save_checkpoint(model,
                              metrics_val['avg_loss'],
                              i,
                              results_dir)
        utils.compute_weights_tree(
            writer, model,
            epoch=i,
        )
        if not is_linear:
            utils.compute_gradients_tree(
                writer, dl_train_small, model, epoch=i)
        i += 1
