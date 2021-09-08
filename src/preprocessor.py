import numpy as np
import torch
import torchaudio


def get_mel_spectrogram(speech_dir_list, max_time=3):
    # max_time (float): maximum time to listen for identifying speaker.
    mel_spectrogram_list = []
    mel_spectrogram_transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=40
    )

    for speech_dir in speech_dir_list:
        waveform, sample_rate = torchaudio.load(speech_dir)
        waveform = torch.mean(waveform, dim=0)

        duration = waveform.shape[0]
        if duration > max_time * sample_rate:
            start_time = int((duration - max_time * sample_rate) / 2)
            end_time = start_time + max_time * sample_rate
            waveform = waveform[start_time : end_time]
        if duration < max_time * sample_rate:
            waveform = torch.cat(
                [waveform, torch.tensor([0 for _ in range(max_time * sample_rate - duration)])],
                dim = 0
            )
        mel_spectrogram = mel_spectrogram_transformer(waveform)
        mel_spectrogram_list.append(np.array(mel_spectrogram))
    return torch.tensor(mel_spectrogram_list)