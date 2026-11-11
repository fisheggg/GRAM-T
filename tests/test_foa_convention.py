"""Self-check for src/data_modules/foa_convention.py. Run: python tests/test_foa_convention.py"""
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


foa = _load("src/data_modules/foa_convention.py")
FeatureExtractor = _load("src/model/ambisonic_feature_extractor.py").FeatureExtractor


def test_plane_wave_from_above():
    # N3D, ch1-up plane wave from straight above: ch1 = sqrt(3) * W.
    w = torch.randn(2, 1000)
    rirs = torch.stack((w, math.sqrt(3) * w, torch.zeros_like(w), torch.zeros_like(w)), dim=1)
    out = foa.match_finetune_foa(rirs)
    assert out.shape == rirs.shape
    torch.testing.assert_close(out[:, 0], w)
    torch.testing.assert_close(out[:, 2], w)  # SN3D Z: unit gain, vertical on channel 2
    assert out[:, 1].abs().max() == 0 and out[:, 3].abs().max() == 0


def test_legacy_adapters_invert_rir_conversion():
    rirs = torch.randn(3, 4, 500)
    out = foa.match_finetune_foa(rirs)
    back = out.clone()
    back[:, 1:4] *= math.sqrt(3.0)  # sn3d_to_n3d, as in cls_feature_class_gram
    torch.testing.assert_close(back[:, [0, 2, 1, 3]], rirs)


def test_feature_swap_equals_waveform_swap():
    extractor = FeatureExtractor(sample_rate=32000, n_fft=1024, win_length=1024, hop_length=320,
                                 f_min=50, f_max=16000, n_mels=128, power=2.0)
    audio = torch.randn(2, 4, 32000)
    feat = extractor(audio).reshape(2, 7, 128, -1)
    feat_of_swapped = extractor(audio[:, [0, 2, 1, 3]]).reshape(2, 7, 128, -1)
    torch.testing.assert_close(foa.swap_vertical_into_ch1(feat), feat_of_swapped)


def test_rejects_wrong_shapes():
    for fn, x in [(foa.match_finetune_foa, torch.zeros(2, 2, 10)),
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
