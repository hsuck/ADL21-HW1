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

        self.rnn = nn.GRU(
            input_size = embeddings.shape[1],
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional,
            batch_first = True, # batch * seq * feature
        )

        for name, param in self.rnn.named_parameters():
            if name.startswith('weight'):
                nn.init.orthogonal_( param )
            else:
                nn.init.zeros_( param )

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Dropout( self.dropout ),
            nn.Linear( in_features = self.encoder_output_size, out_features = self.hidden_size // 2 ),
            nn.BatchNorm1d( self.hidden_size // 2 ),
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
        x = self.embed( batch )
        x, hidden = self.rnn( x, None )
        if self.bidirectional:
            out = self.classifier( torch.sum( x, dim = 1 ) )
        else:
            out = self.classifier( x[:, -1, :] )
        return out
        #raise NotImplementedError


class SeqTagger(SeqClassifier):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super().__init__(
            hidden_size = hidden_size,
            embeddings = embeddings,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = bidirectional,
            num_class = num_class
        )

        #self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        #self.hidden_size = hidden_size
        #self.num_layers = num_layers
        #self.dropout = dropout
        #self.bidirectional = bidirectional
        #self.num_class = num_class

        self.classifier = nn.Sequential(
            nn.PReLU(),
            nn.Dropout( self.dropout ),
            nn.LayerNorm( self.encoder_output_size ),
            #nn.Linear( in_features = self.encoder_output_size, out_features = self.num_class ),
            nn.Linear( in_features = self.encoder_output_size, out_features = self.hidden_size // 2 ),
            nn.PReLU(),
            nn.Dropout( self.dropout ),
            nn.LayerNorm( self.hidden_size // 2 ),
            nn.Linear( in_features = self.hidden_size // 2 , out_features = self.num_class ),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed( batch )
        x, hidden = self.rnn( x )
        out = self.classifier( x )
        return out
        #raise NotImplementedError
