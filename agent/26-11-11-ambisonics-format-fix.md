# 26-11-11 Ambisonics channel-order and normalization fix (branch `fix/ambisonics-format`)

Pretraining RIRs (GRAM-Naturalistic-Scenes-Ambisonics) are N3D with vertical on raw channel 1; STARSS23/DCASE is SN3D, ACN `[W,Y,Z,X]`, vertical on channel 2. Upstream GRAM-T converted neither. Ported the downstream fix from the archived `li_review` branch (`d57e15b`) and Stage 3 code.

**Pretraining is unchanged** (identical to `master`), so it matches upstream and all existing checkpoints. Only SELD converts STARSS to the pretraining convention. A first version also converted the pretraining RIRs (`foa_match_finetune`); that was removed, because it is the alternative fix and combining both would convert twice.

## Switches (`seld-dcase-2022-GRAM/parameters.py`, applied to `dataset='foa'` only)
| Parameter | Default | Effect |
|---|---|---|
| `sn3d_to_n3d` | True | STARSS dipoles × √3 on the waveform before the −14 dBFS RMS normalization; features cached in `foa_dev_n3d` |
| `swap_vertical_to_ch1` | True | feature order `[0,2,1,3,5,4,6]` at the start of `GRAM.forward`; labels stay in STARSS coordinates |

Set both False to reproduce historical SELD runs.

## Files
`seld-dcase-2022-GRAM/foa_convention.py` (new), `seld-dcase-2022-GRAM/{parameters,cls_feature_class_gram,seldnet_model}.py`, `tests/test_foa_convention.py` (new).

## Checks
`python tests/test_foa_convention.py` passes: a STARSS plane wave from above gives exactly the features of the corresponding N3D/ch1-up pretraining plane wave; the feature swap equals the waveform swap through the real feature extractor; bad shapes are rejected. No SELD training run yet.

## Not done
- Horizontal axes (RIR ch2/ch3 vs Y/X) are not matched; that mapping is unresolved and the Habitat hypothesis is untested.
- √3 is applied before RMS normalization (li_review order); Stage 3 applied it after. They are not equivalent.
- `train_seldnet.py` still hard-codes Snellius weight/data paths; SELD has not been ported to Olivia.
