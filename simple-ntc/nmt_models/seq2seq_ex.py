import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self,
                 word_vec_dim,
                 hidden_size,
                 n_layers=4,
                 dropout_p=.2):

        super(Encoder, self).__init__()

        # Be aware of value of 'batch_first' parameter.
        # Also, its hidden_size is half of original hidden_size, because it is bidirectional.

        self.rnn = nn.LSTM(word_vec_dim,
                           int(hidden_size/2),
                           num_layers=n_layers,
                           dropout=dropout_p,
                           bidirectional=True,
                           batch_first=True)


    def forward(self, emb):
        # |emb| = (batch_size, length, word_vec_dim)
        # isinstance : 자료형 확인
        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)