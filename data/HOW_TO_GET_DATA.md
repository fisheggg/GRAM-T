- All data saved to `/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/`.

## STEP 1: get pretraining data
- Pretraining needs three parts of data: AudioSet, WHAM, and GRAM-Naturalistic-Scenes-Ambisonics.
- Use `download_dataset.py` to get the AudioSet and GRAM-naturual-scenes from huggingface.
- For WHAM, first download it from: http://wham.whisper.ai/, then run `pack_wham_data.py`.

## STEP 2: get SELD data
- Final merge of SELD data at: `/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/SELD_combined`
- SELD data contains two parts: STARSS23, and synthetic data from STARSS22.
- Manually download STARSS23 and STARSS22-syntheic from zenodo.
- STARSS data are sampled in 48k, so they need to be converted to 32k.
    - Use the script `convert_starss_sr.py`
- Compute features using `seld-dcase-2022-GRAM/batch_feature_extraction.py`
    - remember to change the directory setting in `seld-dcase-2022-GRAM/parameters.py`
