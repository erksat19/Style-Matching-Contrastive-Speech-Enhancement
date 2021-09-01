import os
import sys
sys.path.append('denoiser')
import subprocess
import yaml
import time
import logging
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import argparse
import shutil

import torch
import torchaudio
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"
from speechbrain.pretrained import EncoderDecoderASR

import dataloader
import evaluate


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


def calculate_wer(asr_model, stt_dict):
    total_len, total_sub, total_ins, total_del = 0, 0, 0, 0
    for speech_dir, ground_truth in stt_dict.items():
        hypothesis = asr_model.transcribe_file(path = speech_dir)
        wer_result_dict = evaluate.wer(ground_truth, hypothesis)

        total_len += len(ground_truth.split())
        total_sub += wer_result_dict["Sub"]
        total_ins += wer_result_dict["Ins"]
        total_del += wer_result_dict["Del"]

        print(total_len, total_sub, total_ins, total_del)

    wer = round((total_sub + total_ins + total_del) / total_len, 3) * 100
    return wer


def denoise(args, stt_dict):
    enhanced_stt_dict = dict()
    model_dir = os.path.abspath(args["denoiser"]["model_dir"])
    noisy_dir = os.path.abspath(args["denoiser"]["noisy_dir"])
    clean_dir = os.path.abspath(args["denoiser"]["clean_dir"])

    print("1")

    shutil.rmtree(noisy_dir, ignore_errors = True)
    shutil.rmtree(clean_dir, ignore_errors = True)
    os.makedirs(noisy_dir)
    os.makedirs(clean_dir)

    print("2")

    for speech_dir, ground_truth in stt_dict.items():
        shutil.copy(speech_dir, noisy_dir)
        enhanced_speech_dir = os.path.abspath(os.path.join(clean_dir, os.path.basename(speech_dir).split(".")[0] + "_enhanced.wav"))
        enhanced_stt_dict[enhanced_speech_dir] = ground_truth

    print("3")
    cur_time = time.time()

    commands = [
         f"python -m denoiser.enhance --model_path={model_dir} --noisy_dir={noisy_dir} --out_dir={clean_dir}"
    ]
    for command in commands:
        result = subprocess.run(
            command,
            cwd = os.path.abspath(args["denoiser"]["repository_dir"]),
            shell = True,
            capture_output = True,
            text = True
        )

    print("4", time.time() - cur_time)

    return enhanced_stt_dict


def main(args):
    for logger_name in ['speechbrain.pretrained.fetching', 'requests.packages.urllib3.connectionpool', 'speechbrain.utils.parameter_transfer']:
        disable_logger(logger_name)

    asr_model = load_asr_model(args["asr"]["source_dir"], args["asr"]["save_dir"])
    stt_dict = dataloader.parse_metadata(args["dataset"]["metadata_dir"])

    print("before denoising")
    if args["denoiser"]["calculate"]:
        stt_dict = denoise(args, stt_dict)
    print("after denoising")

    wer = calculate_wer(asr_model, stt_dict)
    print(wer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, required = False, default = "conf/config.yaml")
    args = parser.parse_args()

    with open(args.config_dir, "r") as config_file:
        args = yaml.load(config_file)
    main(args)