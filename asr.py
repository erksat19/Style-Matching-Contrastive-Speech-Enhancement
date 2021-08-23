import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"

from speechbrain.pretrained import EncoderDecoderASR

import dataloader

def load_asr_model(source_dir, save_dir):
    asr_model = EncoderDecoderASR.from_hparams(
        source = source_dir,
        run_opts = {"device" : device}
    )
    return asr_model

def transcribe(asr_model, speech_dir):
    text = asr_model.transcribe_file(path = speech_dir)
    return text

if __name__ == "__main__":
    # source_dir = "speechbrain/asr-crdnn-rnnlm-librispeech"
    # save_dir = "pretrained_models/asr-crdnn-rnnlm-librispeech"
    source_dir = "pretrained_models/asr-crdnn-rnnlm-librispeech"
    metadata_dir = "/home/data/LibriSpeech/test-other.csv"

    asr_model = load_asr_model(source_dir, save_dir)
    stt_dict = dataloader.parse_metadata(metadata_dir)
    for speech_dir, text in stt_dict.items():
        text = transcribe(asr_model, speech_dir)