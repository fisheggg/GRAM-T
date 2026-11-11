"""Match STARSS23 FOA to the GRAM pretraining convention (downstream only).

Pretraining RIRs (GRAM-Naturalistic-Scenes-Ambisonics) are N3D with the
vertical axis on raw channel 1; pretraining is left unchanged. STARSS23 / DCASE
audio is SN3D in ACN order [W, Y, Z, X], vertical on channel 2 (Z). SELD
therefore converts STARSS:

- ``sn3d_to_n3d``: dipoles x sqrt(3) on the waveform, before RMS normalization.
- ``swap_vertical_to_ch1``: move Z and I_Z into feature channels 1 and 4.

Labels stay in STARSS coordinates. The horizontal mapping of the RIR channels
2/3 is unresolved; only the dipole gain and the vertical axis are matched.
"""

import math

import torch

# Feature stack [W, Y, Z, X, I_Y, I_Z, I_X]: swap (Y, I_Y) with (Z, I_Z).
VERTICAL_TO_CH1 = [0, 2, 1, 3, 5, 4, 6]


def scale_dipoles_sn3d_to_n3d(audio: torch.Tensor) -> torch.Tensor:
    """SN3D -> N3D for (..., 4, time) audio. Apply to the waveform: mel filters mix bins."""
    if audio.shape[-2] != 4:
        raise ValueError(f"expected 4 FOA channels at dim -2, got {tuple(audio.shape)}")
    out = audio.clone()
    out[..., 1:4, :] *= math.sqrt(3.0)
    return out


def swap_vertical_into_ch1(feat: torch.Tensor) -> torch.Tensor:
    """(B, 7, T, F) features -> vertical in channels 1 and 4.

    Exact on features: log-mel is per channel and the IV denominator is symmetric
    in the dipoles. Apply after augmentation, just before the encoder.
    """
    if feat.shape[1] != 7:
        raise ValueError(f"expected 7 feature channels at dim 1, got {tuple(feat.shape)}")
    return feat[:, VERTICAL_TO_CH1]
