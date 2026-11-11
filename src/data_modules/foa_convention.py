"""FOA channel-order and normalization conventions.

Pretraining RIRs (GRAM-Naturalistic-Scenes-Ambisonics) are N3D with the
vertical axis on raw channel 1. STARSS23 / DCASE audio is SN3D in ACN order
[W, Y, Z, X], vertical on channel 2 (Z). Historically neither was converted.

- New pretraining: ``match_finetune_foa`` on the RIRs, so checkpoints match
  STARSS natively (``data.foa_match_finetune``).
- Legacy checkpoints at finetune time: dipoles x sqrt(3) on the STARSS
  waveform (``sn3d_to_n3d``, in cls_feature_class_gram) and ``swap_vertical_into_ch1`` on the features
  (``swap_vertical_to_ch1``). Never combine these with a converted checkpoint.

The horizontal mapping of the RIR channels 2/3 is unresolved; only the vertical
axis and the dipole gain are corrected.
"""

import math

import torch

# Feature stack [W, Y, Z, X, I_Y, I_Z, I_X]: swap (Y, I_Y) with (Z, I_Z).
VERTICAL_TO_CH1 = [0, 2, 1, 3, 5, 4, 6]


def match_finetune_foa(rirs: torch.Tensor) -> torch.Tensor:
    """N3D, ch1-up RIRs -> SN3D, Z-up. ``rirs`` is (..., 4, time)."""
    if rirs.shape[-2] != 4:
        raise ValueError(f"expected 4 FOA channels at dim -2, got {tuple(rirs.shape)}")
    w, c1, c2, c3 = rirs.unbind(-2)
    return torch.stack((w, c2 / math.sqrt(3.0), c1 / math.sqrt(3.0), c3 / math.sqrt(3.0)), dim=-2)


def swap_vertical_into_ch1(feat: torch.Tensor) -> torch.Tensor:
    """Move STARSS Z and I_Z into feature channels 1 and 4. ``feat`` is (B, 7, T, F).

    Exact on features: log-mel is per channel and the IV denominator is symmetric
    in the dipoles. Apply after augmentation; labels stay in STARSS coordinates.
    """
    if feat.shape[1] != 7:
        raise ValueError(f"expected 7 feature channels at dim 1, got {tuple(feat.shape)}")
    return feat[:, VERTICAL_TO_CH1]
