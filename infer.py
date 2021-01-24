import numpy as np
import onnxruntime

from lib import CLIP_LENGTH, load_audio, stft

ONNX_FILE = "gender.onnx"


def infer(wav: "(time,)") -> float:
    clip_starts = range(0, len(wav), CLIP_LENGTH)
    clip_lengths = np.asarray(
        [min(CLIP_LENGTH, len(wav) - start) for start in clip_starts]
    )
    weights = clip_lengths / clip_lengths.sum()
    clips_stft: "(batch, T, F, 2)" = stft_batch(
        fix_length(wav[start:], CLIP_LENGTH) for start in clip_starts
    )
    preds = onnx_run(ONNX_FILE, clips_stft)[:, 0]
    return float(np.sum(weights * preds))


def onnx_run(onnx_file, inp):
    session = onnxruntime.InferenceSession(onnx_file)
    input_name = session.get_inputs()[0].name
    (preds,) = session.run(None, {input_name: inp})
    return preds


def stft_batch(clip: "(batch, time)") -> "(batch, T, F, 2)":
    return np.asarray([stft(batch) for batch in clip])


def fix_length(x: "(..., a)", l: int) -> "(..., l)":
    """Trim or pad `x` to be of length `l` in the last axis."""
    x = x[..., :l]
    return np.pad(x, (*([(0, 0)] * (x.ndim - 1)), (0, l - x.shape[-1])))


if __name__ == "__main__":
    import sys
    files = sys.argv[1:]
    wavs = [load_audio(f) for f in files]
    probas = [infer(w) for w in wavs]
    print("Male probabilities:", file=sys.stderr)
    for f, p in zip(files, probas):
        print(f"{p:.3f} {f}")
