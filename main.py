import linear_lstm_model as lin_model
import dataset
import utils

import torch.nn as nn
import torch
import torch.utils.data as tud
from typing import Union, Tuple, Optional

from tensorboardX import SummaryWriter  # type: ignore

TAKE_TOP10 = False
ds_args = {'validation_proportion': 0.8, 'flatten': True,
           'take_top10': TAKE_TOP10,
           'drop_large_threshold_tokens': 400,
           'max_per_class': 500}
d_train = dataset.Dataset(is_training_not_validation = True,
                          **ds_args)  # type: ignore
d_val = dataset.Dataset(is_training_not_validation = False,
                        **ds_args)  # type: ignore

dl_train = dataset.create_data_loader(d_train, 16)
dl_val = dataset.create_data_loader(d_val, 16)

model = lin_model.LinearLSTM(embedding_dim=20,
                             lstm_output_dim=20,
                             num_classes=10 if TAKE_TOP10 else 104)
print(model)

print(f"Train Dataset length: {len(d_train)}")
print(f"Train Dataloader length (num batches): {len(dl_train)}")
print(f"Val Dataset length: {len(d_val)}")
print(f"Val Dataloader length: {len(dl_val)}")
print(f"Num classes: {10 if TAKE_TOP10 else 104}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train_epoch(model: lin_model.LinearLSTM,
                optimizer: torch.optim.Optimizer,
                train_dl: tud.DataLoader
                ):
    B = utils.batch_size(train_dl).to(model.device)
    
    sum_loss = torch.tensor(0., requires_grad=False).to(model.device)
    num_correct = torch.tensor(0, requires_grad=False).to(model.device)
    
    for batch_idx, (programs, lengths, targets) in enumerate(train_dl):
        optimizer.zero_grad()
        targets = targets.to(model.device)
        logits = model(programs, lengths)
        loss = model.loss(logits, targets)
        metrics = utils.compute_batch_metrics_linear(
            model, logits, loss,
            (programs, lengths, targets))
        num_correct += metrics['num_correct']
        sum_loss += metrics['loss'] * B
        loss.backward()
        optimizer.step()
        #if batch_idx % 100 == 0:
        #    print(torch.mean(loss))
    num_batches = torch.tensor(float(batch_idx)).to(model.device)
    avg_loss = sum_loss / num_batches
    avg_accuracy = num_correct / (num_batches * B)
    return {'avg_loss': avg_loss, 'avg_accuracy': avg_accuracy}
    
NUM_EPOCHS = 200
RESULTS_DIR = utils.create_model_directory(ds_args)
writer = SummaryWriter(RESULTS_DIR)
for i in range(NUM_EPOCHS):
    print(f"\nEPOCH: {i+1}/{NUM_EPOCHS}:")
    metrics_train = train_epoch(model, optimizer, dl_train)
    writer.add_scalar('train/avg_loss', metrics_train['avg_loss'], i)
    writer.add_scalar('train/avg_accuracy', metrics_train['avg_accuracy'], i)
    
    metrics_val = utils.validation_metrics_linear(model, dl_val)
    writer.add_scalar('val/avg_loss', metrics_val['avg_loss'], i)
    writer.add_scalar('val/avg_accuracy', metrics_val['avg_accuracy'], i)
    
    print(f"train loss: {metrics_train['avg_loss']:.3f}")
    print(f"train accuracy: {metrics_train['avg_accuracy']*100:.2f}")
    print(f"val loss: {metrics_val['avg_loss']:.3f}")
    print(f"val accuracy: {metrics_val['avg_accuracy']*100:.2f}")
    utils.save_checkpoint(model, metrics_val['avg_loss'], i, RESULTS_DIR)
