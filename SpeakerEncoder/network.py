import torch
import torch.nn as nn
# import print_structure
import tensorboardX

import SpeakerEncoder.hparams as hp


class SpeakerEncoder(nn.Module):
    """Speaker Encoder"""

    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        self.lstm = nn.LSTM(hp.n_mels_channel,
                            hp.hidden_dim,
                            num_layers=hp.num_layer,
                            batch_first=True)
        self.projection = nn.Linear(hp.hidden_dim, hp.speaker_dim)
        self.init_params()

    def init_params(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        # ============================== #
        # input: (batch, length, n_mels) #
        # ============================== #

        x, _ = self.lstm(x)
        x = x[:, x.size(1) - 1]
        x = self.projection(x)
        x = x / torch.norm(x)

        return x


if __name__ == "__main__":

    test_model = SpeakerEncoder()
    print(test_model)

    input_x = torch.randn(2, 180, 80)
    y = test_model(input_x)
    # print(y.size())
    # dot = print_structure.make_dot(y)
    # dot.render('test.gv', view=True)
    # print(dot)
    # print(type(dot))
    # dot.view()

    # with tensorboardX.SummaryWriter(comment='test') as w:
    #     w.add_graph(test_model, (input_x,))
