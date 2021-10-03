import os
import math
from time import gmtime, strftime
import argparse
import yaml
import gc

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

import dataloader
import preprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Use {device} as device")


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)


class StyleEncoder(nn.Module):
    """
    Style encoder is a speaker identification model for style transfer.
    Args:
        - input_dim (int): input dimension of LSTM, which is the dimension of mel-spectrogram.
        - hidden_dim (int): hidden dimension of LSTM.
        - projection_dim (int): projection dimension style encoder.
        - output_dim (int): the number of speakers.
        - depth (int): number of layers.
        - floor (float): stability flooring when normalizing.
    """
    def __init__(self,
                 input_dim=40,
                 hidden_dim=512,
                 projection_dim=256,
                 output_dim=10,
                 depth=3,
                 floor=1e-4):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.output_dim = output_dim
        self.depth = depth
        self.floor = floor

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=depth).to(device)
        self.projection_layer = nn.Linear(
            in_features=hidden_dim,
            out_features=projection_dim).to(device)
        self.final_layer = nn.Linear(
            in_features=projection_dim,
            out_features=output_dim).to(device)
        self.activation = Swish().to(device)

    def forward(self, mel_spectrogram):
        # mel_spectrogram: (batch_size, n_mels, seq_len)
        mean = mel_spectrogram.mean(dim=1, keepdim=True)
        std = mel_spectrogram.std(dim=1, keepdim=True)
        normalized_mel_spectrogram = (mel_spectrogram - mean) / (self.floor + std)

        # batch
        normalized_mel_spectrogram = normalized_mel_spectrogram.permute(2, 0, 1) # (seq_len, batch_size, n_mels)
        out, _ = self.lstm(normalized_mel_spectrogram)
        out = out[-1, :, :]
        out = self.activation(self.projection_layer(out))
        out = self.final_layer(out)
        return out

    def get_style_vector(self, mel_spectrogram):
        mean = mel_spectrogram.mean(dim=1, keepdim=True)
        std = mel_spectrogram.std(dim=1, keepdim=True)
        normalized_mel_spectrogram = (mel_spectrogram - mean) / (self.floor + std)

        # batch
        normalized_mel_spectrogram = normalized_mel_spectrogram.permute(2, 0, 1) # (seq_len, batch_size, n_mels)
        out, _ = self.lstm(normalized_mel_spectrogram)
        out = out[-1, :, :]
        out = self.projection_layer(out)
        return out

def save_model(model, out_dir, epoch, loss, val_loss):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f"weights_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.pt")
    torch.save(model.state_dict(), out_path)


def plot(history, out_dir, training_start_time):
    plt.subplot(2, 1, 1)
    plt.title('accuracy vs epoch')
    plt.plot(history['accuracy'], 'orange')
    plt.plot(history['val_accuracy'], 'green')
    plt.legend(['accuracy', 'val_accuracy'], loc = 'upper right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.subplot(2, 1, 2)
    plt.title('loss vs epoch')
    plt.plot(history['loss'], 'orange')
    plt.plot(history['val_loss'], 'green')
    plt.legend(['loss', 'val_loss'], loc = 'upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"training_result_{training_start_time}.png"))


def train(args):
    dataset = dataloader.StyleEncoderDataset(args)
    train_size = int(len(dataset) * args["style_encoder"]["train_ratio"])
    validation_size = len(dataset) - train_size
    train_set, validation_set = torch.utils.data.random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_set, batch_size=args["style_encoder"]["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_set, batch_size=args["style_encoder"]["batch_size"], shuffle=False)

    model = StyleEncoder(output_dim=dataset.num_speakers).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=float(args["style_encoder"]["lr"]))
    epochs = args["style_encoder"]["epochs"]
    history = {"loss" : [], "val_loss" : [], "accuracy" : [], "val_accuracy" : []}
    training_start_time = strftime("%Y_%m_%d_%H:%M:%S", gmtime())

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        epoch_train_size, epoch_validation_size = 0, 0
        epoch_train_loss, epoch_validation_loss = 0, 0
        epoch_train_correct, epoch_validation_correct = 0, 0

        # train
        for item in tqdm(train_dataloader):
            model.train()
            speech_dir, speaker_id = item["speech_dir"], item["speaker_id"].to(device)
            mel_spectrogram = preprocessor.get_mel_spectrogram_from_speech_dir(speech_dir).to(device)

            output_distribution = model(mel_spectrogram).to(device)

            loss = loss_function(output_distribution, speaker_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_size += len(speech_dir)
            epoch_train_loss += loss.cpu().detach().numpy()
            epoch_train_correct += int(torch.sum(torch.argmax(output_distribution.data, -1) == speaker_id))

            gc.collect()
            if "cuda" in device:
                torch.cuda.empty_cache()

        # validation
        for item in tqdm(validation_dataloader):
            model.eval()
            speech_dir, speaker_id = item["speech_dir"], item["speaker_id"].to(device)
            mel_spectrogram = preprocessor.get_mel_spectrogram_from_speech_dir(speech_dir).to(device)
            output_distribution = model(mel_spectrogram).to(device)

            loss = loss_function(output_distribution, speaker_id)

            epoch_validation_size += len(speech_dir)
            epoch_validation_loss += loss.cpu().detach().numpy()
            epoch_validation_correct += int(torch.sum(torch.argmax(output_distribution.data, -1) == speaker_id))

            gc.collect()
            if "cuda" in device:
                torch.cuda.empty_cache()

        epoch_train_loss /= epoch_train_size
        epoch_validation_loss /= epoch_validation_size
        epoch_train_correct /= epoch_train_size
        epoch_validation_correct /= epoch_validation_size

        history["loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_validation_loss)
        history["accuracy"].append(epoch_train_correct)
        history["val_accuracy"].append(epoch_validation_correct)

        print(f"loss : {epoch_train_loss:.6f}, val_loss : {epoch_validation_loss:.6f}, accuracy : {epoch_train_correct:.4f}, val_accuracy : {epoch_validation_correct:.4f}")
        save_model(model, args["style_encoder"]["out_dir"], epoch, epoch_train_loss, epoch_validation_loss)
        plot(history, args["style_encoder"]["out_dir"], training_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=False, default="conf/config.yaml")
    args = parser.parse_args()

    with open(args.config_dir) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    train(config)