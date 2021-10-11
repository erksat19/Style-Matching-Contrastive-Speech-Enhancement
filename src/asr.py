import os
import sys
sys.path.append('denoiser')
import subprocess
import yaml
import time
from datetime import datetime
import logging
import warnings
warnings.filterwarnings("ignore")
import json
import argparse
import shutil

import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"
import torchaudio
from speechbrain.pretrained import EncoderDecoderASR

import dataloader
import evaluate



def disable_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)


def load_asr_model(source_dir, save_dir):
    asr_model = EncoderDecoderASR.from_hparams(
        source=source_dir,
        savedir=save_dir,
        run_opts={"device" : device}
    )
    return asr_model


def calculate_wer(asr_model, stt_dict):
    total_len, total_sub, total_ins, total_del = 0, 0, 0, 0
    for speech_dir, ground_truth in stt_dict.items():
        hypothesis = asr_model.transcribe_file(path=speech_dir)

        # remove symbolic link created by transcribe_file function
        sym_link = os.path.split(speech_dir)[-1]
        if os.path.exists(sym_link):
            os.remove(sym_link)

        # print(ground_truth)
        # print(hypothesis)

        wer_result_dict = evaluate.wer(ground_truth, hypothesis)

        total_len += wer_result_dict["Cor"] + wer_result_dict["Sub"] + wer_result_dict["Del"]
        total_sub += wer_result_dict["Sub"]
        total_ins += wer_result_dict["Ins"]
        total_del += wer_result_dict["Del"]

        # print(f"total_len : {total_len}, total_sub : {total_sub}, total_ins : {total_ins}, total_del : {total_del}")

    wer = round((total_sub + total_ins + total_del) / total_len, 3) * 100
    return wer


def denoise(args, stt_dict):
    enhanced_stt_dict = dict()
    pretrained_model_dir = args["denoiser"]["pretrained_model_dir"]
    model_dir = os.path.abspath(args["denoiser"]["model_dir"])
    noisy_dir = os.path.abspath(args["denoiser"]["noisy_dir"])
    clean_dir = os.path.abspath(args["denoiser"]["clean_dir"])

    shutil.rmtree(noisy_dir, ignore_errors=False, onerror=None)
    shutil.rmtree(clean_dir, ignore_errors=False, onerror=None)
    os.makedirs(noisy_dir)
    os.makedirs(clean_dir)

    for speech_dir, ground_truth in stt_dict.items():
        shutil.copy(speech_dir, noisy_dir)
        enhanced_speech_dir = os.path.abspath(os.path.join(clean_dir, os.path.basename(speech_dir).split(".")[0] + "_enhanced.wav"))
        enhanced_stt_dict[enhanced_speech_dir] = ground_truth

    if args["denoiser"]["use_pretrained_model"]:
        commands = [
            f"{sys.executable} -m denoiser.enhance --{pretrained_model_dir} --noisy_dir={noisy_dir} --out_dir={clean_dir}"
        ]
    else:
        commands = [
            f"{sys.executable} -m denoiser.enhance --model_path={model_dir} --noisy_dir={noisy_dir} --out_dir={clean_dir}"
        ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=os.path.abspath(args["denoiser"]["repository_dir"]),
            shell=True,
            capture_output=True,
            text=True
        )
    return enhanced_stt_dict


def generate_noisy_speech(clean_speech, target_snr): # generate gaussian noise
    if len(clean_speech.shape) > 1:
        clean_speech = torch.mean(clean_speech, dim=0)
    noise = torch.normal(mean=0, std=1.0, size=clean_speech.shape)

    rms_clean_speech = torch.sqrt(torch.mean(torch.square(clean_speech)))
    rms_noise = torch.sqrt(torch.mean(torch.square(noise)))
    current_snr = 20 * torch.log10(rms_clean_speech / rms_noise)

    adjustment_constnat = 10 ** ((current_snr - target_snr) / 20)
    noise *= adjustment_constnat
    noisy_speech = clean_speech + noise
    if torch.max(torch.abs(noisy_speech)) >= 32767: # adjust maximum value
        noise_speech *= 32767 / torch.max(torch.abs(noisy_speech))
    return noisy_speech


def generate_noisy_speech_from_speech_dir(stt_dict, root_dir, target_snr):
    noisy_stt_dict = dict()

    out_dir = os.path.join(root_dir, "SNR" + str(target_snr))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for speech_dir, ground_truth in stt_dict.items():
        clean_speech, sample_rate = torchaudio.load(speech_dir)
        noisy_speech = generate_noisy_speech(clean_speech, target_snr)
        file_dir = os.path.join(out_dir, os.path.split(speech_dir)[-1])
        torchaudio.save(file_dir, noisy_speech.unsqueeze(0), sample_rate)
        noisy_stt_dict[file_dir] = ground_truth
    return noisy_stt_dict


def do_wer_curve_experiment(args):
    experimental_result = dict()
    stt_dict = dataloader.get_stt_dict(args["dataset"]["metadata_dir"])
    asr_model = load_asr_model(args["asr"]["source_dir"], args["asr"]["save_dir"])

    out_dir = os.path.join(*[
        args["wer_curve_experiment"]["root_dir"],
        datetime.now().strftime("out_%Y_%m_%d_%H_%_M_%S.txt")
    ])

    f = open(out_dir, "w")
    f.write("Hyperparameter Info" + "\n" + "=" * 30 + "\n")
    f.write(json.dumps(args, indent=2) + "\n\n")
    f.write("SNR    Noisy_WER   Enhanced_WER\n" +"=" * 30 + "\n")
    f.close()

    for target_snr in ["original"] + args["wer_curve_experiment"]["snr_list"]:
        if target_snr == "original":
            noisy_stt_dict = stt_dict
        else:
            noisy_stt_dict = generate_noisy_speech_from_speech_dir(stt_dict, args["wer_curve_experiment"]["root_dir"], target_snr)

        enhanced_stt_dict = denoise(args, noisy_stt_dict)
        noisy_wer = calculate_wer(asr_model, noisy_stt_dict)
        enhanced_wer = calculate_wer(asr_model, enhanced_stt_dict)

        f = open(out_dir, "a")
        f.write(f"{str(target_snr)}    {noisy_wer}    {enhanced_wer}\n")
        f.close()

        experimental_result["SNR" + str(target_snr)] = {
            "noisy": noisy_wer,
            "enhanced": enhanced_wer
        }

    return experimental_result


def main(args):
    for logger_name in ['speechbrain.pretrained.fetching', 'requests.packages.urllib3.connectionpool', 'speechbrain.utils.parameter_transfer']:
        disable_logger(logger_name)
    experimental_result = do_wer_curve_experiment(args)
    # print(experimental_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=False, default="conf/config.yaml")
    args = parser.parse_args()

    with open(args.config_dir, "r") as config_file:
        args = yaml.load(config_file)
    main(args)