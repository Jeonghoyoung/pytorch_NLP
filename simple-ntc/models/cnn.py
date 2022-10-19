import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(
            self,
            input_size,
            word_vec_size,
            n_classes,
            use_batch_norm=False,
            dropout_p =0.5,
            window_sizes=[3,4,5],
            n_filters=[100, 100, 100],
    ):
        self.input_size = input_size,
        self.word_vec_size = word_vec_size,
        self.n_classes = n_classes,
        self.use_batch_norm = use_batch_norm,
        self.dropout_p = dropout_p,
        # window_size means that how many words a pattern covers
        self.window_sizes = window_sizes,
        # n_filters means that how many patterns to cover
        self.n_filters = n_filters,

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size)
        # Use nn.ModuleList to register each sub-modules.
        self.feature_extractors = nn.ModuleList()
        for window_size, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1, # we only use one embedding layer
                        out_channels=n_filter,
                        kernel_size=(window_size, word_vec_size),
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(dropout_p),
                )
            )
        # An input of generator layer is max values from each filter.
        self.generator = nn.Linear(sum(n_filters), n_classes)

        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)

        # |x| = (batch_size, length, word_vec_dim)
        min_length = max(self.window_sizes)
        if min_length > x.size(1):
            # Because some input does not long enough for maximum length of window size
            # we add zero tensor for padding
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vec_size).zero_()
            # |pad| = (batch_size, min_length - length, word_vec_dim)
            x = torch.cat([x, pad], dim=-1)
            # |x| = (batch_size, min_length, word_vec_dim)

        # In ordinary case of vision task, you may have 3 channels in tensor,
        # but in this case, you would have just 1 channel,
        # which is added by 'unsqueeze' method in below:
        x = x.unsqueeze(1)
        # |x| = (batch_size, 1, length, word_vec_dim)
        cnn_outs = []

        for block in self.feature_extractors:
            cnn_out = block(x)

            cnn_out = nn.functional.max_pool1d(
                input = cnn_out.squeeze(-1),
                kernel_size=cnn_out.size(-2)
            ).squeeze(-1)

            cnn_outs += [cnn_out]
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        y = self.activation(self.generator(cnn_outs))
        return y