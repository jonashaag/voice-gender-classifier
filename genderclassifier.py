import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm.contrib.concurrent

from lib import CLIP_LENGTH, MIN_CLIP_LENGTH, get_mel_mat, load_audio, stft

EPS = 1e-20
# This is a random list of Common Voice languages.
VAL_LANGS = "fy-NL vot ia hu ky sv-SE hsb ga-IE sl cv en or ta ka hi".split()


def load_random_clip(for_val, rng=np.random.default_rng(), _ignore=set()):
    """Get a random clip suitable for training.

    - Must be of at least MIN_CLIP_LENGTH
    - Draw female/male with same probability

    _ignore: Set of files that don't have any clips that are long enough.
    """
    metadata = (
        (cv_val_female, cv_val_male) if for_val else (cv_train_female, cv_train_male)
    )
    metadata = metadata[rng.integers(2)]
    while True:
        name, age, gender, lang = metadata[rng.integers(len(metadata))]
        if name in _ignore:
            continue
        path = CV_ROOT.joinpath(lang, "clips", name).with_suffix(CV_FILES_SUFFIX)
        try:
            wav = load_audio(path)
        except Exception as err:
            print(f"Error reading {path}: {err}, ignoring.", file=sys.stderr)
            _ignore.add(name)
            continue
        wav /= np.abs(wav).max() + EPS
        nonsilent_clips = list(
            filter(
                lambda clip: clip[1] - clip[0] > MIN_CLIP_LENGTH,
                librosa.effects.split(wav, 35),
            )
        )
        if nonsilent_clips:
            start, end = nonsilent_clips[rng.integers(len(nonsilent_clips))]
            return wav[start:end], (name, age, gender)
        else:
            _ignore.add(name)


def fix_length_random(rng, arr, target_length):
    """Randomly trim or pad `arr` so that is of length `target_length`."""
    diff = target_length - len(arr)
    if diff > 0:
        pad_left = rng.integers(diff)
        return np.pad(arr, (pad_left, diff - pad_left))
    elif diff < 0:
        off = rng.integers(-diff)
        return arr[off:][:target_length]
    else:
        return arr


class GenderDataset(torch.utils.data.Dataset):
    def __init__(self, for_val: bool):
        self.for_val = for_val

    def __len__(self):
        return 100_000 if self.for_val else 1_000_000

    def __getitem__(self, idx):
        # Use deterministic randomness for validation.
        rng = np.random.default_rng(idx if self.for_val else torch.seed())
        wav, (name, age, gender) = load_random_clip(self.for_val, rng=rng)
        wav = fix_length_random(rng, wav, CLIP_LENGTH)
        # Volume augmentation.
        wav *= rng.uniform(0.1, 1)
        return (
            torch.from_numpy(stft(wav)),
            np.asarray([0 if gender == "female" else 1]).astype("float32"),
        )


def ConvBnReLU2d(in_chan, out_chan, kernel_size, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_chan, out_chan, kernel_size, stride),
        torch.nn.BatchNorm2d(out_chan),
        torch.nn.ReLU(),
    )


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = ConvBnReLU2d(1, 16, (5, 3), (2, 1))
        self.l2 = ConvBnReLU2d(16, 32, (5, 3), (2, 1))
        self.l3 = ConvBnReLU2d(32, 64, (3, 5), (1, 2))
        self.l4 = ConvBnReLU2d(64, 64, (3, 3), (1, 1))
        self.l5 = ConvBnReLU2d(64, 16, (3, 3), (1, 1))
        self.out = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(16 * 23 * 23, 1)
        )
        self.register_buffer("mel", torch.from_numpy(get_mel_mat()))

    def forward(self, x):
        # Amplitude to magnitude STFT.
        x = (x[..., 0] ** 2 + x[..., 1] ** 2).sqrt()
        # Magnitude STFT to Mel spectrogram.
        x = self.mel.matmul(x.transpose(1, 2).abs())
        x = x.unsqueeze(1)
        for l in [self.l1, self.l2, self.l3, self.l4, self.l5]:
            x = l(x)
        x = self.out(x)
        x = x.sigmoid()
        return x


class ClassifierTraining(pl.LightningModule):
    LR = 1e-5
    BATCH_SIZE = 128
    NUM_WORKERS = 12

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.accuracy = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        return self.common_step("t", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.common_step("v", batch, batch_idx)

    def common_step(self, t, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        self.log(f"{t}_acc", self.accuracy(yhat, y), prog_bar=True)
        return torch.nn.functional.binary_cross_entropy(yhat, y)

    def val_epoch_end(self, outs):
        self.log("v_acc_ep", self.accuracy.compute(), prog_bar=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            GenderDataset(False),
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            GenderDataset(True),
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=False,
        )

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.parameters(), lr=self.LR),
        ]


def parse_cv_metadata(f):
    """Parse Common Voice .tsv files"""
    df = pd.read_csv(f, sep="\t")[["path", "age", "gender"]]
    df["lang"] = f.parent.name
    return df


def create_cv_splits(cv_root):
    """Read Common Vocie metadata and create female/male train/validation splits."""
    available_files = list(cv_root.glob(f"*/clips/*{CV_FILES_SUFFIX}"))
    assert len(available_files) > 100, "Incorrect Common Voice paths?"
    cv_metadata = pd.concat(
        tqdm.contrib.concurrent.process_map(
            parse_cv_metadata, list(cv_root.glob("*/validated.tsv"))
        ),
        copy=False,
    )
    cv_metadata = cv_metadata[
        cv_metadata["path"].isin({f.with_suffix(".mp3").name for f in available_files})
    ]
    # Drop other and missing genders
    cv_metadata = cv_metadata.replace("other").dropna(subset=["gender"])
    # fmt: off
    cv_female       = cv_metadata[cv_metadata["gender"] == "female"]
    cv_train_female = cv_female[~cv_female["lang"].isin(VAL_LANGS)].to_numpy()
    cv_val_female   = cv_female[cv_female["lang"].isin(VAL_LANGS)].to_numpy()
    cv_male       = cv_metadata[cv_metadata["gender"] == "male"]
    cv_train_male = cv_male[~cv_male["lang"].isin(VAL_LANGS)].to_numpy()
    cv_val_male   = cv_male[cv_male["lang"].isin(VAL_LANGS)].to_numpy()
    # fmt: on
    print(
        "Number of female and male entries",
        len(cv_train_female),
        len(cv_train_male),
        len(cv_val_female),
        len(cv_val_male),
    )
    return cv_train_female, cv_train_male, cv_val_female, cv_val_male


if __name__ == "__main__":
    # Model quick check
    dummy_input = torch.from_numpy(stft(np.random.uniform(size=(CLIP_LENGTH,)))[None])
    model = Classifier()
    model(dummy_input)

    # Change to your Common Voice folder.
    # It is recommended to decode the Common Voice MP3s to 16 kHz WAV files before training.
    CV_ROOT = Path("cv-corpus-6.1-2020-12-11")
    CV_FILES_SUFFIX = ".wav"
    cv_train_female, cv_train_male, cv_val_female, cv_val_male = create_cv_splits(
        CV_ROOT
    )

    from pytorch_lightning.callbacks import ModelCheckpoint

    trainer = pl.Trainer(
        weights_summary="full",
        callbacks=[ModelCheckpoint(dirpath="ckpts", save_last=True)],
        gpus=1,
        max_epochs=5,
    )
    trainer.fit(ClassifierTraining(model))
    torch.onnx.export(
        model,
        dummy_input,
        "gender.onnx",
        opset_version=11,
        dynamic_axes={"0": {0: "batch"}},
    )
