"""Self-check for seld-dcase-2022-GRAM/foa_convention.py. Run: python tests/test_foa_convention.py"""
import importlib.util
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(rel):
    # Load by path: the package __init__ files import Lightning, which is not needed here.
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


foa = _load("seld-dcase-2022-GRAM/foa_convention.py")
FeatureExtractor = _load("src/model/ambisonic_feature_extractor.py").FeatureExtractor
extractor = FeatureExtractor(sample_rate=32000, n_fft=1024, win_length=1024, hop_length=320,
                             f_min=50, f_max=16000, n_mels=128, power=2.0)


def features(audio):
    return extractor(audio).reshape(audio.shape[0], 7, 128, -1)


def downstream(starss):
    """SELD path: waveform SN3D -> N3D, features, then vertical into ch1."""
    return foa.swap_vertical_into_ch1(features(foa.scale_dipoles_sn3d_to_n3d(starss)))


def test_plane_wave_from_above_matches_pretraining():
    w = torch.randn(1, 32000)
    zero = torch.zeros_like(w)
    starss = torch.stack((w, zero, w, zero), dim=1)                 # SN3D, Z = W
    pretrain = torch.stack((w, math.sqrt(3) * w, zero, zero), dim=1)  # N3D, ch1 = sqrt(3) W
    torch.testing.assert_close(downstream(starss), features(pretrain))


def test_feature_swap_equals_waveform_swap():
    audio = torch.randn(2, 4, 32000)
    n3d = foa.scale_dipoles_sn3d_to_n3d(audio)
    torch.testing.assert_close(downstream(audio), features(n3d[:, [0, 2, 1, 3]]))


def test_rejects_wrong_shapes():
    for fn, x in [(foa.scale_dipoles_sn3d_to_n3d, torch.zeros(2, 10)),
                  (foa.swap_vertical_into_ch1, torch.zeros(2, 4, 10, 10))]:
        try:
            fn(x)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted shape {tuple(x.shape)}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
