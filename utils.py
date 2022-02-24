import linear_lstm_model as lin_model

import torch
import torch.utils.data as tud
from typing import Union, Tuple, Optional, Dict, Any
import pathlib
import datetime
from tensorboardX import SummaryWriter  # type: ignore

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

def create_model_directory(ds_args: Dict[str, Any]) -> str:
    now = datetime.datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H--%M--%S")

    model_kind = 'linear' if ds_args['flatten'] else 'tree'
    classes_size = '10' if ds_args['take_top10'] else '104'
    p = pathlib.Path(__file__).resolve().parent
    results_dir = p / f'results/model_{model_kind}__classes_{classes_size}/{timestamp}'
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
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
