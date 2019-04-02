import torch
import os
import numpy as np

import network
import SpeakerEncoder
import text
import hparams as hp
import audio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(model_Tacotron2, SpeakerEncoder, text, ref_mel):
    with torch.no_grad():
        embedding = SpeakerEncoder(ref_mel)
        output = model_Tacotron2.inference(text, embedding)
        # print(output[1].size())
        output = output[1].squeeze()
        # print(output.size())

    return output


if __name__ == "__main__":
    # Test
    model_tacotron2 = network.Tacotron2(hp).to(device)
    checkpoint_path = "checkpoint_148200.pth.tar"
    checkpoint_path = os.path.join("batch_big", checkpoint_path)
    model_tacotron2.load_state_dict(torch.load(
        os.path.join(hp.checkpoint_path, checkpoint_path))['model'])
    model_tacotron2.eval()
    Speaker_Encoder = SpeakerEncoder.get_model().to(device)

    test_wav = audio.load_wav("test.wav")
    mel_spec = audio.melspectrogram(test_wav)
    # print(np.shape(mel_spec))
    mel_spec = np.transpose(mel_spec)[0:180]
    # print(np.shape(mel_spec))
    mel_spec = torch.from_numpy(mel_spec).float().to(device)
    mel_spec = torch.stack([mel_spec])
    # print(mel_spec.size())

    test_text = "What can you do?"
    test_text = text.text_to_sequence(test_text, hp.text_cleaners)
    test_text = torch.Tensor(test_text).long().to(device)
    test_text = torch.stack([test_text])
    # print(test_text.size())

    output = test(model_tacotron2, Speaker_Encoder, test_text, mel_spec)

    output = output.cpu().numpy()
    wav_results = audio.inv_mel_spectrogram(output)
    audio.save_wav(wav_results, "result.wav")
