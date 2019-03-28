import torch
from torch.utils.data import Dataset, DataLoader
from text import text_to_sequence
import hparams as hp
import utils

from nnmnkwii.datasets import vctk
import numpy as np
from multiprocessing import cpu_count
import os

device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


class Tacotron2DataLoader(Dataset):
    """VCTK"""

    def __init__(self):
        self.vctk_path = hp.vctk_path
        self.dataset_path = hp.dataset_path
        self.speakers = vctk.available_speakers
        self.list_mel_spec = os.listdir(hp.dataset_path)
        self.text = self.get_text()

    def get_text(self):
        td = vctk.TranscriptionDataSource(hp.vctk_path, speakers=self.speakers)
        transcriptions = td.collect_files()
        return transcriptions

    def __len__(self):
        return len(self.list_mel_spec)

    def __getitem__(self, index):
        mel_spec_name = os.path.join(
            self.dataset_path, self.list_mel_spec[index])
        mel_np = np.load(mel_spec_name).T
        # print(np.shape(mel_np))

        character = self.text[index]
        character = text_to_sequence(character, hp.text_cleaners)
        character = np.array(character)

        # print(mel_np)
        return {"text": character, "mel": mel_np}


def collate_fn(batch):

    texts = [d['text'] for d in batch]
    mels = [d['mel'] for d in batch]

    length_text = np.array(list())
    for text in texts:
        length_text = np.append(length_text, np.shape(text)[0])

    length_mel = np.array(list())
    for mel in mels:
        length_mel = np.append(length_mel, np.shape(mel)[0])

    texts = pad_seq_text(texts)
    mels = pad_seq_spec(mels)

    index = np.argsort(- length_text)
    new_texts = np.stack([texts[i] for i in index])
    length_text = np.stack([length_text[i]
                            for i in np.argsort(- length_text)]).astype(np.int32)

    index = np.argsort(- length_mel)

    new_SE_mels = np.stack([mels[i] for i in index])
    new_SE_mels = utils.cut_for_SpeakerEncoder(new_SE_mels)
    # print(np.shape(new_SE_mels))
    new_mels = np.stack([np.transpose(mels[i])for i in index])
    length_mel = np.stack([length_mel[i]
                           for i in np.argsort(- length_mel)]).astype(np.int32)

    total_len = np.shape(new_mels)[2]
    gate_padded = np.stack([gen_gate(total_len, length_mel[i])
                            for i in range(np.shape(new_mels)[0])])

    # =========================================================== #
    # new_texts: (batch, maxlen)                                  #
    # length_text: (longest_len, the second longest lenght, ...)  #
    # new_mels: (batch, mel_len, maxlen)                          #
    # gate_padded: (batch, maxlen_mel)                            #
    # length_mel: (longest_len, the second longest lenght, ...)   #
    # mel_for_SE: (batch, hp.tisv_frame, hp.num_mels)             #
    # =========================================================== #

    # print(new_SE_mels)
    # print(new_mels)
    return new_texts, length_text, new_mels, gate_padded, length_mel, new_SE_mels


def gen_gate(total_len, target_len):
    # print(target_len)
    # print(total_len)
    out = np.array([0 for i in range(total_len)])
    for i in range(target_len-1, total_len):
        out[i] = 1

    return out


def pad_seq_text(inputs):
    def pad_data(x, length):
        pad = 0
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=pad)

    max_len = max((len(x)for x in inputs))
    return np.stack([pad_data(x, max_len)for x in inputs])


def pad_seq_spec(inputs):
    def pad(x, max_len):
        # print(type(x))
        if np.shape(x)[0] > max_len:
            # print("ERROR!")
            raise ValueError("not max_len")
        s = np.shape(x)[1]
        # print(s)
        x = np.pad(x, (0, max_len - np.shape(x)
                       [0]), mode='constant', constant_values=0)
        return x[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    return np.stack([pad(x, max_len)for x in inputs])

# def process_data_for_SpeakerEncoder(mel_specs):


if __name__ == "__main__":

    dataset = Tacotron2DataLoader()
    train_loader = DataLoader(dataset, num_workers=cpu_count(
    ), shuffle=True, batch_size=hp.batch_size, drop_last=True, collate_fn=collate_fn)

    print("total length of train loader:", len(train_loader))
    for i, batch in enumerate(train_loader):
        if i == 1:
            break

        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, mel_for_SE = batch
        print("text:", np.shape(text_padded))
        print("input length:", np.shape(input_lengths))
        # print(input_lengths)
        print("mel:", np.shape(mel_padded))
        print("gate:", np.shape(gate_padded))
        print("mel length:", output_lengths)
        # print(gate_padded)
        print("mel SE:", np.shape(mel_for_SE))

        # cnt = 0
        # for i in gate_padded[0]:
        #     if i == 1:
        #         cnt = cnt + 1
        # print(cnt)

        # cnt = 0
        # for i in gate_padded[1]:
        #     if i == 1:
        #         cnt = cnt + 1
        # print(cnt)

    # text_a = np.array([1, 2, 3, 4, 5, 6])
    # text_b = np.array([1, 2, 3])

    # mel_spec_a = np.ones((2, 2))
    # mel_spec_b = np.ones((3, 2))

    # a = pad_seq_text([text_a, text_b])
    # b = pad_seq_spec([mel_spec_a, mel_spec_b])
    # c = gen_gate(12, 10)

    # print(a)
    # print(b)
    # print(c)
