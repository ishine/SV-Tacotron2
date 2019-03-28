import numpy as np
from scipy.io.wavfile import read
import torch
import hparams as hp


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def pad_zeros(mel_specs):
    temp_mel_spec = mel_specs.copy()
    padded_zeros = np.zeros((hp.tisv_frame, hp.num_mels))
    # print(np.shape(padded_zeros))
    return np.stack([np.concatenate((batch, padded_zeros), 0) for ind, batch in enumerate(temp_mel_spec)])


def cut_for_SpeakerEncoder(mel_specs):
    if np.shape(mel_specs)[1] < hp.tisv_frame:
        # print("open")
        padded_mel_specs = pad_zeros(mel_specs)
        temp_list = list()
        for batch in padded_mel_specs:
            temp = batch[0:hp.tisv_frame, :]
            temp_list.append(temp)

        return np.stack(temp_list)
    else:
        temp_list = list()
        for batch in mel_specs:
            temp = batch[0:hp.tisv_frame, :]
            temp_list.append(temp)

        return np.stack(temp_list)


def add_encoder_embedding(encoder_outputs, embeddings):
    for ind in range(encoder_outputs.size(0)):
        for index in range(encoder_outputs.size(1)):
            encoder_outputs[ind][index] = encoder_outputs[ind][index] + \
                embeddings[ind]
    return encoder_outputs


if __name__ == "__main__":

    # out = get_mask_from_lengths(torch.arange(
    #     0, 12, out=torch.cuda.LongTensor(12)))
    # print(~out)

    mel_test = np.ones((2, 100, 80))
    a = cut_for_SpeakerEncoder(mel_test)
    # print(np.shape(a))
    # print(np.shape(mel_test))
    # print(a)

    e_o = torch.ones(2, 2, 3)
    e_s = torch.zeros(2, 3)
    a = add_encoder_embedding(e_o, e_s)
    print(a)
