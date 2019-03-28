from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable

import network
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visual():
    model = network.Tacotron2(hp).to(device)
    inputs = torch.ones(2, 50).long().to(device)
    input_length = torch.Tensor([50, 50]).long().to(device)
    targets = torch.randn(2, 80, 200).to(device)
    max_len = torch.randn(2,3).to(device)  # Can't be a value???
    output_lengths = torch.Tensor([200, 200]).long().to(device)
    embeddings = torch.randn(2, 256).to(device)
    inputs_all = inputs, input_length, targets, max_len, output_lengths

    model(inputs_all, embeddings)
    print("**********************")
    with SummaryWriter(comment='SV-Tacotron2') as w:
        w.add_graph(model, (inputs_all, embeddings))


if __name__ == "__main__":
    # Test
    visual()
