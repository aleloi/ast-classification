import dataset
import linear_lstm_model as lin_model
# import tree_lstm_model as tree_model
import dgl_lstm_model as dgl_model
import collections

import numpy as np
import torch
import torch.utils.data as tud
from typing import Union, Tuple, Optional, Dict, Any, List
import pathlib
import datetime
import dgl  # type: ignore
from tensorboardX import SummaryWriter  # type: ignore
    
def compute_batch_metrics_tree(model: dgl_model.DGLTreeLSTM,
                               logits: Optional[torch.Tensor],
                               loss: Optional[torch.Tensor],
                               batch: Tuple[dgl.DGLGraph, torch.Tensor]
                               ):
    progs = dgl.batch([prog for prog, _ in batch]).to(model.device)
    targets = torch.stack([cls for _, cls in batch]).to(model.device)
    with torch.no_grad():
        if logits is None:
            logits = model(progs)
        if loss is None:
            loss = model.loss(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        return {'num_correct': torch.sum(predictions == targets),
                'loss': loss}


def compute_batch_metrics_linear(model: lin_model.LinearLSTM,
                                 logits: Optional[torch.Tensor],
                                 loss: Optional[torch.Tensor],
                                 batch: Tuple[torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor]
                                 ):
    
    programs, lengths, targets = [x.to(model.device) for x in batch]
    lengths = lengths.to(torch.device('cpu'))
    with torch.no_grad():
        if logits is None:
            logits = model(programs, lengths)
        if loss is None:
            loss = model.loss(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        return {'num_correct': torch.sum(predictions == targets),
                'loss': loss}
    
def batch_size(dl: tud.DataLoader) -> torch.Tensor:
    BATCH_SIZE = dl.batch_size if dl.batch_size is not None else -1
    assert BATCH_SIZE != -1
    return torch.tensor(float(BATCH_SIZE))

def validation_metrics_linear(model: lin_model.LinearLSTM,
                       val_dl: tud.DataLoader):
    CPU = torch.device('cpu')
    B = batch_size(val_dl).to(model.device)

    sum_loss = torch.tensor(0., requires_grad=False).to(model.device)
    num_correct = torch.tensor(0, requires_grad=False).to(model.device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dl):
            metrics = compute_batch_metrics_linear(model, None, None, batch)
            num_correct += metrics['num_correct']
            sum_loss += metrics['loss'] * B

        num_batches = torch.tensor(float(batch_idx)).to(model.device)
        avg_loss = sum_loss / num_batches
        avg_accuracy = num_correct / (num_batches * B)
        return {'avg_loss': float(avg_loss),
                'avg_accuracy': float(avg_accuracy)}

def validation_metrics_tree(model: dgl_model.DGLTreeLSTM,
                            val_dl: tud.DataLoader):
    CPU = torch.device('cpu')
    B = batch_size(val_dl).to(model.device)

    sum_loss = torch.tensor(0., requires_grad=False).to(model.device)
    num_correct = torch.tensor(0, requires_grad=False).to(model.device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dl):
            metrics = compute_batch_metrics_tree(model, None, None, batch)
            num_correct += metrics['num_correct']
            sum_loss += metrics['loss'] * B

        num_batches = torch.tensor(float(batch_idx)).to(model.device)
        avg_loss = sum_loss / num_batches
        avg_accuracy = num_correct / (num_batches * B)
        return {'avg_loss': float(avg_loss),
                'avg_accuracy': float(avg_accuracy)}

def create_model_directory(ds_args: dataset.DataArgs,
                           lr: float,
                           model: lin_model.LinearLSTM,
                           really_create = True,
                           include_timestamp = True,
                           ) -> str:
    now = datetime.datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H--%M--%S")

    model_kind = 'linear' if ds_args.flatten else 'tree'
    classes_size = '10' if ds_args.take_top10 else '104'
    p = pathlib.Path(__file__).resolve().parent
    results_dir = p / f'results/'
    float_to_str = lambda f: str(f).replace('.', '_')
    model_name = (f'model_{model_kind}__classes_{classes_size}' 
                  f'__emb_dim_{model.embedding_dim}' 
                  f'__lstm_dim_{model.lstm_output_dim}' 
                  f'__fc_depth_{len(model.fc_layers)}' 
                  f'__label_smoothing_{float_to_str(model.label_smoothing)}'
                  f'__lr_{float_to_str(lr)}'
                  )

    
    results_dir = results_dir / model_name
    if include_timestamp:
         results_dir = results_dir / f'{timestamp}'
    if really_create:
        pathlib.Path(results_dir).mkdir(
            parents=True, exist_ok=True)
    return str(results_dir)


def save_checkpoint(model, val_loss, epoch, save_dir):
    print(f"\t\t >>> epoch is {epoch}. Saving model ... <<< ")
    best_model_filename = f'{save_dir}/Epoch--{epoch}-Loss--{val_loss:.2f}.pt'
    torch.save(model.state_dict(), best_model_filename)
        
if __name__ == '__main__':
    ds_args = {'validation_proportion': 0.8, 'flatten': True,
               'take_top10': True,
               'drop_large_threshold_tokens': 200,
               'max_per_class': 1000}
    #print(create_model_directory(ds_args))

def last_modified_subdir(path):
    p = pathlib.Path(path)
    dir_p = [x for x in p.iterdir() if x.is_dir()]
    return max(dir_p, key=lambda x: x.stat().st_mtime)

def load_model(model, save_dir, epoch=None):
    """Loads most recent model from 'save_dir'"""

    p = pathlib.Path(save_dir)
    assert p.parts[-1].startswith("model_"), p

    print(f"Loading model from {save_dir}...")
    dir_p = [x for x in p.iterdir() if x.is_dir()]
    runs = [x.parts[-1] for x in dir_p]
    print(f"{save_dir} contains runs {' '.join(runs)}")
    latest_run = last_modified_subdir(p) 
    print(f"Latest run is: {latest_run.parts[-1]}")

    epochs = [x for x in latest_run.iterdir()
              if x.parts[-1].endswith(".pt")]
    num_epochs = len(epochs)
    if epoch is not None:
        epochs = [x for x in epochs if f"Epoch--{epoch}-" in x.parts[-1]]
    latest_epoch = max(epochs,
                       key=lambda x: x.stat().st_mtime)
    print(f"Loading epoch: {latest_epoch.parts[-1]} out of {num_epochs} epochs")
    

    kwargs = {}
    if torch.cuda.device_count() == 0:
        kwargs['map_location'] = torch.device('cpu')
    state_dict = torch.load(latest_epoch, **kwargs)
    model.load_state_dict(state_dict)
    return latest_run
    
    
def compute_embeddings(writer, model,
                       model_name,
                       epoch=None,
                       save_dir=None,
                       do_load=True):
    INT_TO_TOKEN = {i: tok for (tok, i) in
                    dataset.TOKEN_TO_INT.items()}
    tok_names = [INT_TO_TOKEN[i]
                 for i in range(len(INT_TO_TOKEN))]
    tok_names.append('PAD_TOKEN')

    if do_load:
        assert save_dir is not None
        load_model(model, save_dir, epoch=None)
    writer.add_embedding(model.embedding.weight,
                         metadata=tok_names,
                         # save dir is
                         # 'results / <model_name> / <timestamp> / <save_file>.pt'
                         # we want 'model name'
                         #save_dir.parent.parent.parts[-1]
                         tag=model_name,
                         global_step=epoch
                         )


def compute_gradients_tree(
        writer, dataset, model,
        save_dir = None, epoch=None):
    """
    Send in a sub-dataset (should be part of the training data if the
    goal is to estimate training gradients.
    """
    assert save_dir is not None or epoch is not None
    def log_gradients(epoch):
        l2s = collections.defaultdict(list)
        raw_grads = collections.defaultdict(list)
        MAX_RAW_GRADS_PER_EPOCH=100000
        for batch in dataset:
            progs = dgl.batch([prog for prog, _ in batch]).to(model.device)
            targets = torch.stack([cls for _, cls in batch]).to(model.device)
            logits_t = model(progs) 
            loss = model.train_loss(logits_t, targets)
            loss.backward()
            for (name, param) in model.named_weights():
                assert param.grad is not None
                l2_grad = torch.linalg.norm(param.grad)
                # default is Frobenius norm
                l2s[name].append(l2_grad)

                num_raw_grads = 0
                if len(raw_grads[name]) > 0:
                    num_raw_grads = len(raw_grads[name])*len(raw_grads[name][0])
                if num_raw_grads < MAX_RAW_GRADS_PER_EPOCH:
                    raw_grads[name].append(param.grad.flatten())
                    
        for key in l2s:
            writer.add_histogram(
                f'train/{key}_l2_grad',
                torch.stack(l2s[key]).cpu().numpy(),
                epoch)
            writer.add_histogram(
                f'train/{key}_raw_grad',
                torch.stack(raw_grads[key]).cpu().numpy(),
                epoch)
    if epoch is not None:
        log_gradients(epoch)
    else:
        # TODO fix 'while True'?
        epoch = 0
        while True:
            load_model(model, save_dir, epoch)
            log_gradients()
            epoch += 1
        
def compute_weights_tree(writer, model,
                         save_dir=None,
                         epoch=None
                         ):
    """
    Send in a sub-dataset (should be part of the training data if the
    goal is to estimate training gradients.
    """
    def write_one(epoch):
        for (name, param) in model.named_weights():
            assert param.grad is not None
            with torch.no_grad():
                writer.add_histogram(
                    f'train/{name}_weight_values',
                    param.flatten().cpu().numpy(),
                    epoch)
    if epoch is None:
        assert save_dir is not None
        epoch = 0
        while True:
            load_model(model, save_dir, epoch)
            write_one(epoch)
            epoch += 1
    else:
        assert save_dir is None
        write_one(epoch)

        


