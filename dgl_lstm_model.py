import dataset

from typing import Tuple, Optional, List
import torch.nn as nn
import torch
import dgl  # type: ignore
from copy import deepcopy


class DGLTreeLSTM(nn.Module):
    def __init__(self, embedding_dim: int,
                 lstm_output_dim : int,
                 num_classes: int,
                 extra_non_linear: Optional[int],
                 label_smoothing: Optional[float]):
        super(DGLTreeLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_output_dim = lstm_output_dim
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        # One learnable vector for every token type.
        self.embedding = nn.Embedding(
            num_embeddings=dataset.PAD_TOKEN_VALUE+1,
            embedding_dim=embedding_dim,
            padding_idx=dataset.PAD_TOKEN_VALUE)

        self.lstm_cell = ChildSumTreeLSTMCell(embedding_dim,
                                              lstm_output_dim)

        # Output is logit.
        linear_output_size = (extra_non_linear if
                              extra_non_linear is not None
                              else num_classes)
        self.linear = nn.Linear(lstm_output_dim,
                                linear_output_size)
        fc_layers : List[nn.Module] = [self.linear]
        if extra_non_linear is not None:
            fc_layers.append(nn.Tanh())
            fc_layers.append(nn.Linear(linear_output_size,
                                       num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.loss = nn.CrossEntropyLoss()
        if self.label_smoothing is not None:
            self.train_loss = nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing)
        else: 
            self.train_loss = self.loss

        # Initialization is done by default, because param matrices
        # are nn.Linear layers.
        # self.device = torch.device("cpu") # works faster for some reason.
        self.set_gpu_use()


    def named_weights(self):
        res = [
                    ('embedding', self.embedding.weight),
                    ('W_iou', self.lstm_cell.W_iou.weight),
                    ('U_iou', self.lstm_cell.U_iou.weight),
                    ('b_iou', self.lstm_cell.b_iou),
                    ('U_f', self.lstm_cell.U_f.weight),
                    ('b_f', self.lstm_cell.U_f.bias),
                    ('linear_1_w', self.fc_layers[0].weight),
                    ('linear_1_b', self.fc_layers[0].bias)
            ]
        if len(self.fc_layers) == 3:
            for p in [
                    ('linear_2_w', self.fc_layers[2].weight),
                    ('linear_2_b', self.fc_layers[2].bias)
            ]:
                res.append(p)
        return res
        

    def set_gpu_use(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device [{self.device}].')
        assert torch.cuda.is_available()
        self.to(self.device)

    def forward(self, batch: dgl.DGLGraph):
        batch.ndata['x_vec'] = self.embedding(batch.ndata['x'])
        (tot_toks,) = batch.ndata['x'].size()
        batch.ndata['h'] = torch.zeros(
            tot_toks,
            self.lstm_output_dim).to(self.device)
        batch.ndata['c'] = torch.zeros(
            tot_toks,
            self.lstm_output_dim).to(self.device)
        batch.ndata['iou'] = self.lstm_cell.W_iou(batch.ndata['x_vec'])

        dgl.prop_nodes_topo(batch,
                            self.lstm_cell.message_func,
                            self.lstm_cell.reduce_func,
                            apply_node_func=self.lstm_cell.apply_node_func)

        graphs = dgl.unbatch(batch)
        h_outputs = torch.stack(
            [graph.ndata['h'][0] for graph in graphs])
        return self.fc_layers(h_outputs)

# Adapted from the DGL example repository.
class ChildSumTreeLSTMCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}
