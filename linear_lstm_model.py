from typing import Tuple
import torch.nn as nn
import torch

import dataset


class LinearLSTM(nn.Module):
    def __init__(self, embedding_dim: int,
                 lstm_output_dim : int,
                 num_classes: int
                 ):
        super(LinearLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_output_dim = lstm_output_dim
        self.num_classes = num_classes
        
        # One learnable vector for every token type.
        self.embedding = nn.Embedding(
            num_embeddings=dataset.PAD_TOKEN_VALUE+1,
            embedding_dim=embedding_dim,
            padding_idx=dataset.PAD_TOKEN_VALUE)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_output_dim)

        # Output is logit.
        self.linear = nn.Linear(lstm_output_dim, num_classes)

        # Default settings is batch average and softmax of inputs.
        self.loss = nn.CrossEntropyLoss()

        self.set_gpu_use()
        

    def forward(self,
                progs: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """progs is padded sequences. It is T x B where T is the max length
        in the batch.
        
        length is a 1D tensor of sequences lengths.

        Returns B x self.num_classes - dimensional output.

        """
        progs = progs.to(self.device)

        T, B = progs.size()

        progs_embed = self.embedding(progs)
        assert progs_embed.size() == (T, B, self.embedding_dim)

        progs_packed = torch.nn.utils.rnn.pack_padded_sequence(
            progs_embed, lengths,
            enforce_sorted=False
        )

        # An alternative is to pass (h0, c0) to self.lstm. The default
        # is (h0, c0)=(0, 0). One idea is to initialize them
        # randomly. Is that ingesting noise?
        lstm_output, (h_n, c_n) = self.lstm(progs_packed)
        h_n = torch.squeeze(h_n, 0)
        logits = self.linear(h_n)
        assert logits.size() == (B, self.num_classes), logits.size()

        return logits

    def set_gpu_use(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device [{self.device}].')
        assert torch.cuda.is_available()
        self.to(self.device)
