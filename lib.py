import os

import numpy as np

try:
    from pyfftw.builders import rfft as rfft_builder
except ImportError:
    rfft_builder = lambda *args, **kwargs: np.fft.rfft
import soundfile as sf

try:
    import librosa
except ModuleNotFoundError:
    librosa = None
import julius
import torch

# We work on 2 s 16 kHz clips with at least 1.5 s non-silence.
SAMPLE_RATE = 16_000
MIN_CLIP_LENGTH = int(0.5 * SAMPLE_RATE)
CLIP_LENGTH = int(2.0 * SAMPLE_RATE)


def load_audio(f):
    if librosa:
        wav, sr = librosa.load(f, sr=None, dtype="float32")
    else:
        wav, sr = sf.read(f, dtype="float32", always_2d=True)
        if wav.ndim > 1:
            wav = wav[:, 0]
    return resample(wav, sr, SAMPLE_RATE)


def get_mel_mat():
    if not os.path.exists("mel.npy"):
        import librosa

        np.save("mel", librosa.filters.mel(16000, 512, 128, 50), False)
    return np.load("mel.npy")


def resample(wav_arr, from_sr, to_sr, device="cpu", _resamplers={}):
    assert 1000 <= from_sr <= 96000
    assert 1000 <= to_sr <= 96000
    if from_sr == to_sr:
        return wav_arr
    try:
        resampler = _resamplers[(from_sr, to_sr)]
    except KeyError:
        resampler = _resamplers[(from_sr, to_sr)] = julius.resample.ResampleFrac(
            from_sr, to_sr
        )
    with torch.no_grad():
        return (
            resampler.to(device)(torch.from_numpy(wav_arr.astype("float32")).to(device))
            .cpu()
            .numpy()
            .astype(wav_arr.dtype)
        )


# Optimized STFT implementation from https://gist.github.com/f0k/0e50729c169114fdc6cb41be013a499f
win = np.hanning(512).astype("float32")
n_fft = 512
hop = 512
rfft50 = rfft_builder(np.empty((50, n_fft), "float32"), n=n_fft, threads=1)
rfft1 = rfft_builder(np.empty((1, n_fft), "float32"), n=n_fft, threads=1)


def stft(samples: "(time,)", batch=50) -> "(T, F, 2)":
    num_frames = max(0, (len(samples) - n_fft) // hop + 1)
    rfft = rfft1 if batch == 1 else rfft50
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(num_frames, n_fft),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    )
    res = [
        stack_complex(rfft(frames[pos : pos + batch] * win))
        for pos in range(0, num_frames - batch + 1, batch)
    ]
    if num_frames % batch:
        res.append(stft(samples[(num_frames // batch * batch) * hop :], batch=1))
    return np.vstack(res)


def stack_complex(arr: "(...)") -> "(..., 2)":
    """Stack real and imag of `arr` at new last axis."""
    return np.stack([np.real(arr), np.imag(arr)], axis=-1)
