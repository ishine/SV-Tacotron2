import SpeakerEncoder.network as network
import SpeakerEncoder.hparams as hp

import torch
import torch.nn as nn
import os


def get_model():
    model = network.SpeakerEncoder()
    model.eval()
    # print("Model Have Been Defined")
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_2500.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    # print("Model Have Been Loaded")

    return model


# def get_embedding(mel_data, model):
#     embeddings = model(mel_data)
#     return embeddings
