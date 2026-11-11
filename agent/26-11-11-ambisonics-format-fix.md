# 26-11-11 Ambisonics channel-order and normalization fix (branch `fix/ambisonics-format`)

Pretraining RIRs (GRAM-Naturalistic-Scenes-Ambisonics) are N3D with vertical on raw channel 1; STARSS23/DCASE is SN3D, ACN `[W,Y,Z,X]`, vertical on channel 2. Upstream GRAM-T converted neither. Ported the fix from the archived `li_review` branch (`d57e15b`) and Stage 3 code.

## Switches
| Where | Parameter | Default | Effect |
|---|---|---|---|
| Pretraining, `configs/data/audioset_ambisonics.yaml` | `foa_match_finetune` | **True** | RIRs → SN3D, Z-up at load: `(W, C2/√3, C1/√3, C3/√3)` for source and noise RIRs |
| SELD, `parameters.py` | `sn3d_to_n3d` | False | STARSS dipoles × √3 before −14 dBFS RMS normalization; features cached in `foa_dev_n3d` |
| SELD, `parameters.py` | `swap_vertical_to_ch1` | False | feature order `[0,2,1,3,5,4,6]` at the start of `GRAM.forward` |

New checkpoints match STARSS directly, so both SELD switches stay off for them. The SELD switches exist only for historical N3D/ch1-up checkpoints. Converted checkpoints get `FOA=sn3dZ` in their save path, and `seldnet_model.GRAM` raises if either SELD switch is used with such a checkpoint (report 3 §2.4(a)).

## Files
`src/data_modules/foa_convention.py` (new), `WebAudioDataModule.py`, `train.py`, `utils.py`, `configs/data/audioset_ambisonics.yaml`, `seld-dcase-2022-GRAM/{parameters,cls_feature_class_gram,seldnet_model}.py`, `tests/test_foa_convention.py` (new).

## Checks
`python tests/test_foa_convention.py` passes: plane-wave gain/axis, legacy adapters invert the RIR conversion, feature swap = waveform swap through the real feature extractor, shape rejection. Real shard samples are `(6, 4, 96000)` float64 (source + 5 noise RIRs). Hydra resolves the flag and identity suffix. No training run yet; AudioSet download still in progress.

## Not done
- Horizontal axes (C2/C3 → Y/X) are assigned by convention, not proven; the Habitat mapping is untested.
- `sn3d_to_n3d` is applied before RMS normalization (li_review order); Stage 3 applied it after. They are not equivalent.
- SPR pseudolabels and the archived DOA/rotation helpers were not ported (not in upstream).
