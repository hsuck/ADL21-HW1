from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        # TODO: model architecture

        self.rnn = nn.LSTM(
            input_size = embeddings.shape[1],
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional,
            batch_first = True, # batch * seq * feature
        )

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Dropout( self.dropout ),
            nn.Linear( in_features = self.encoder_output_size, out_features = self.hidden_size // 2 ),
            nn.BatchNorm1d( hidden_size // 2 ),
            nn.SiLU(),
            nn.Dropout( self.dropout ),
            nn.Linear( in_features = self.hidden_size // 2 , out_features = self.num_class ),
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size if not self.bidirectional else 2 * self.hidden_size
        #raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #h0 = torch.zeros( self.num_layers * 2, batch.size(0), self.hidden_size ).to( device )
        #c0 = torch.zeros( self.num_layers * 2, batch.size(0), self.hidden_size ).to( device )
        x = self.embed( batch )
        #out, hidden = self.rnn( x, ( h0, c0 ) )
        out, hidden = self.rnn( x, None )
        if self.bidirectional:
            out = self.classifier( torch.mean( out, dim = 1 ) )
        else:
            out = self.classifier( out[:, -1, :] )
        return out
        #raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
