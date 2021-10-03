import numpy as np
import torch
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_mel_spectrogram_from_speech_dir(speech_dir_list):
    # max_time (float): maximum time to listen for identifying speaker.
    mel_spectrogram_list = []
    for speech_dir in speech_dir_list:
        waveform, sample_rate = torchaudio.load(speech_dir)
        mel_spectrogram = get_mel_spectrogram(waveform, sample_rate)
        mel_spectrogram_list.append(mel_spectrogram)
    return torch.stack(mel_spectrogram_list)


def get_mel_spectrogram_from_waveform_batch(waveform_batch):
    mel_spectrogram_list = []
    for waveform in waveform_batch:
        mel_spectrogram = get_mel_spectrogram(waveform)
        mel_spectrogram_list.append(mel_spectrogram)
    return torch.stack(mel_spectrogram_list)


def get_mel_spectrogram(waveform, sample_rate=16000, max_time=3):
    mel_spectrogram_transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=40
    ).to(device)
    waveform = torch.mean(waveform, dim=0).to(device)
    duration = waveform.shape[0]
    if duration > max_time * sample_rate:
        start_time = int((duration - max_time * sample_rate) / 2)
        end_time = start_time + max_time * sample_rate
        waveform = waveform[start_time : end_time]
    else: # duration < max_time * sample_rate
        waveform = torch.cat(
            [waveform, torch.tensor([0 for _ in range(max_time * sample_rate - duration)]).to(device)],
            dim = 0
        )
    mel_spectrogram = mel_spectrogram_transformer(waveform)
    return mel_spectrogram