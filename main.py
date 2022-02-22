import linear_lstm_model as lin_model
import dataset
import torch.nn as nn
import torch
import torch.utils.data as tud

d = dataset.Dataset(validation_proportion=0.1,
                    flatten=True,
                    take_top10=True,
                    drop_large_threshold_tokens=1000)
dl = dataset.create_data_loader(d, 12)

model = lin_model.LinearLSTM(embedding_dim=20,
                             lstm_output_dim=20,
                             num_classes=10)
print(model)

print(f"Dataset length: {len(d)}")
print(f"Dataloader length: {len(dl)}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_epoch(model: lin_model.LinearLSTM,
                optimizer: torch.optim.Optimizer,
                data_loader: tud.DataLoader                
                ):
    
    for batch_idx, (programs, lengths, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        targets = targets.to(model.device)
        loss = model.loss(model(programs, lengths), targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(torch.mean(loss))


train_epoch(model, optimizer, dl)
