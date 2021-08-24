import os
import sys
sys.path.append('denoiser')
import subprocess
import logging
import warnings
warnings.filterwarnings("ignore")
from functools import partial

import torch
import torchaudio
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"

from speechbrain.pretrained import EncoderDecoderASR
import hydra

import dataloader
import evaluate
import denoiser.enhance


def disable_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)


def load_asr_model(source_dir, save_dir):
    asr_model = EncoderDecoderASR.from_hparams(
        source = source_dir,
        savedir = save_dir,
        run_opts = {"device" : device}
    )
    return asr_model


def calculate_wer(asr_model, stt_dict, denoiser = None):
    total_len, total_sub, total_ins, total_del = 0, 0, 0, 0
    for speech_dir, ground_truth in stt_dict.items():
        if denoiser:
            speech_dir = denoiser(speech_dir = speech_dir)
            # TODO : remove enhance speech
        hypothesis = asr_model.transcribe_file(path = speech_dir)
        wer_result_dict = evaluate.wer(ground_truth, hypothesis)

        total_len += len(ground_truth.split())
        total_sub += wer_result_dict["Sub"]
        total_ins += wer_result_dict["Ins"]
        total_del += wer_result_dict["Del"]

    wer = round((total_sub + total_ins + total_del) / total_len, 3) * 100
    return wer


def denoise(model_dir, speech_dir):
    base_dir = os.path.dirname(speech_dir)
    noisy_dir = os.path.join(base_dir, "noisy")
    clean_dir = os.path.join(base_dir, "clean")
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    password = '8bfsb6'

    commands = [
        f"echo {password}",
        f"sudo -S cp {speech_dir} {noisy_dir}",
        f"python -m denoiser.denoiser.enhance --dns48 --noisy_dir={noisy_dir} --out_dir={clean_dir}"
        # f"python -m denoiser.denoiser.enhance --model_path={model_dir} --noisy_dir={noisy_dir} --out_dir={clean_dir}"
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd = hydra.utils.get_original_cwd(),
            shell = True,
            capture_output = True,
            text = True
        )
        print(result)
    return clean_dir


@hydra.main(config_path = "../conf/config.yaml")
def main(args):
    for logger_name in ['speechbrain.pretrained.fetching', 'requests.packages.urllib3.connectionpool', 'speechbrain.utils.parameter_transfer']:
        disable_logger(logger_name)

    asr_model = load_asr_model(args.source_dir, args.save_dir)
    stt_dict = dataloader.parse_metadata(args.metadata_dir)
    denoiser = partial(denoise, model_dir = args.denoising_model_dir)
    wer = calculate_wer(asr_model, stt_dict, denoiser)
    print(wer)


if __name__ == "__main__":
    main()