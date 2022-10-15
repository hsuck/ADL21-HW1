from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn

RNN = { 'GRU': nn.GRU,
        'LSTM': nn.LSTM }
class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        model: str,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.model = model.split('_')
        #self.model = model
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.CNN = False
        self.num_cnn = 1
        # TODO: model architecture

        # model architecture
        print( self.model )
        if self.model[0] == "CNN":
            self.model.pop(0)
            self.CNN = True

        self.cnn = []
        for i in range( self.num_cnn ):
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels = self.embed.embedding_dim,
                    out_channels = self.embed.embedding_dim,
                    kernel_size = 5,
                    stride = 1,
                    padding = 2, 
                    padding_mode = 'zeros',
                ),
                nn.ReLU(),
                nn.Dropout( self.dropout ),
                nn.BatchNorm1d( self.embed.embedding_dim )
            )
            self.cnn.append( conv )
        self.cnn = nn.ModuleList( self.cnn )

        print( "Using " + model + ' model' )
        self.rnn = RNN[self.model[0]](
        #self.rnn = RNN[self.model](
            input_size = self.embed.embedding_dim,
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
            nn.ReLU(),
            nn.Dropout( self.dropout ),
            nn.Linear( in_features = self.encoder_output_size, out_features = self.hidden_size // 2 ),
            nn.BatchNorm1d( self.hidden_size // 2 ),
            nn.ReLU(),
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

        if self.CNN:
            x = x.permute( 0, 2, 1 )
            for conv in self.cnn:
                x = conv( x )
            x = x.permute( 0, 2, 1 )
            
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
        model: str,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super().__init__(
            model = model,
            hidden_size = hidden_size,
            embeddings = embeddings,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = bidirectional,
            num_class = num_class
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout( self.dropout ),
            nn.LayerNorm( self.encoder_output_size ),
            nn.Linear( in_features = self.encoder_output_size, out_features = self.hidden_size // 2 ),
            #nn.Linear( in_features = self.encoder_output_size, out_features = self.num_class ),
            nn.ReLU(),
            nn.Dropout( self.dropout ),
            nn.LayerNorm( self.hidden_size // 2 ),
            nn.Linear( in_features = self.hidden_size // 2 , out_features = self.num_class ),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #print( batch.shape )
        x = self.embed( batch )
        #print( x.shape )
        if self.CNN:
            x = x.permute( 0, 2, 1 )
            for conv in self.cnn:
                x = conv( x )
            x = x.permute( 0, 2, 1 )

        x, hidden = self.rnn( x )
        #print( x.shape )
        out = self.classifier( x )
        #print( out.shape )
        return out
        #raise NotImplementedError
