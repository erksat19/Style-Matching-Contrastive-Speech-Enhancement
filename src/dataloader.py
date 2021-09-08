import os
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_stt_dict(metadata_dir):
    stt_dict = dict()
    with open(metadata_dir) as metadata_file:
        csv_reader = csv.reader(metadata_file)
        next(csv_reader) # omit first line
        for row in csv_reader: # ID, duration, wav, spk_id, wrd
            stt_dict[row[2]] = row[4]
    return stt_dict


def get_sts_dict(metadata_dir): # speech to speaker
    sts_dict = dict()
    with open(metadata_dir) as metadata_file:
        csv_reader = csv.reader(metadata_file)
        next(csv_reader) # omit first line
        for row in csv_reader: # ID, duration, wav, spk_id, wrd
            sts_dict[row[2]] = row[3].split("-")[0]
    return sts_dict


class StyleEncoderDataset(Dataset):
    def __init__(self, args):
        self.sts_dict = get_sts_dict(args["style_encoder"]["metadata_dir"])
        self.speakers = list(set(self.sts_dict.values()))
        self.num_speakers = len(self.speakers)
        self.input_list, self.output_list = [], []

        for speech_dir, speaker_id in self.sts_dict.items():
            self.input_list.append(speech_dir)
            self.output_list.append(self.speakers.index(speaker_id))
        self.output_list = torch.tensor(self.output_list)

    def __len__(self):
        return len(self.sts_dict)

    def __getitem__(self, idx):
        item = {
            "speech_dir" : self.input_list[idx],
            "speaker_id" : self.output_list[idx]
        }
        return item