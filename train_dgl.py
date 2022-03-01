import dgl_lstm_model as dgl_model
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

TAKE_TOP10 = False
num_classes = 10 if TAKE_TOP10 else 104
BATCH_SIZE=16
LEARNING_RATE = 0.0001

ds_args = dataset.DataArgs(
    flatten=False,
    take_top10=TAKE_TOP10,
    training_weight = 8,
    validation_weight = 2,
    test_weight = 1,
    train_samples = 500*num_classes,
    val_samples = 50*num_classes,
    test_samples = 500*num_classes,
    small_train_samples = 100,
    batch_size=BATCH_SIZE,
    drop_large_threshold_tokens=400
)
dl_train, dl_train_small, dl_val, dl_test = dataset.get_datasets(ds_args)
print(f"Small dataset has {len(dl_train_small)} batches")

model = dgl_model.DGLTreeLSTM(embedding_dim=40,
                              lstm_output_dim=200,
                              num_classes=num_classes,
                              extra_non_linear=200,
                              label_smoothing=0.05
                             )
EPOCH=2
RESULTS_DIR = utils.load_model(
    model,
    'results/' +
    'model_tree__classes_104__emb_dim_40__lstm_dim_200__fc_depth_3__label_smoothing_0_05__lr_0_0001',
    epoch=EPOCH)
print(model)

tot_num_batches = len(dl_train)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_epoch(model, #: tree_model.DGLTreeLSTM,
                optimizer: torch.optim.Optimizer,
                train_dl: tud.DataLoader
                ):
    B = utils.batch_size(train_dl).to(model.device)
    
    sum_loss = torch.tensor(0., requires_grad=False).to(model.device)
    num_correct = torch.tensor(0, requires_grad=False).to(model.device)

    training_start = datetime.datetime.now()
    tot_num_samples = 0
    for batch_idx, batch in enumerate(train_dl):
        batch_training_start = datetime.datetime.now()
        optimizer.zero_grad()
        progs = dgl.batch([prog for prog, _ in batch]).to(model.device)
        targets = torch.stack([cls for _, cls in batch]).to(model.device)
        logits_t = model(progs) 
        loss = model.train_loss(logits_t, targets)
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
            print(f"Statistics for batch {batch_idx+1} of {tot_num_batches}:")
            samples_this_batch = float(B * (batch_idx+1))
            print(f"\t Average loss this epoch: {float(sum_loss / (batch_idx+1))}")
            print(f"\t Average accuracy this epoch: {float(num_correct / samples_this_batch)}")
            now = datetime.datetime.now()
            Δ = now - training_start
            print(f"\t Time per sample (this batch): {Δ / samples_this_batch}")
            print(f"\t Samples per second (this batch): {samples_this_batch / max(Δ.seconds, 1)}")
        
    num_batches = torch.tensor(float(batch_idx)).to(model.device)
    avg_loss = sum_loss / num_batches
    avg_accuracy = num_correct / (num_batches * B)
    return {'avg_loss': avg_loss, 'avg_accuracy': avg_accuracy}
    
NUM_EPOCHS = 200
RESULTS_DIR = utils.create_model_directory(ds_args, LEARNING_RATE, model)
writer = SummaryWriter(RESULTS_DIR)

i = EPOCH+1
while i < NUM_EPOCHS:
    print(f"\nEPOCH: {i+1}/{NUM_EPOCHS}:")
    metrics_train = train_epoch(model, optimizer, dl_train)
    writer.add_scalar('train/avg_loss',
                      metrics_train['avg_loss'], i)
    writer.add_scalar('train/avg_accuracy',
                      metrics_train['avg_accuracy'], i)
    
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
        pathlib.Path(RESULTS_DIR).parent.parts[-1],
        epoch=i,
        do_load=False
        )
    utils.save_checkpoint(model,
                          metrics_val['avg_loss'],
                          i,
                          RESULTS_DIR)
    utils.compute_weights_tree(
        writer, model,
        epoch=i,
        )
    utils.compute_gradients_tree(
        writer, dl_train_small, model, epoch=i)
    i += 1
