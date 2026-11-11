"""
This script is used to convert the STARSS23 and SELD_synth data to 32000 Hz.
And then merge the data into a single directory.

The data structure is as follows:
- SELD_combined
    - foa_eval
    - foa_dev
        - dev-train-sony
        - dev-train-tau
        - dev-test-sony
        - dev-test-tau
        - dev-train-synth
    - metadata_dev
        - dev-train-sony
        - dev-train-tau
        - dev-test-sony
        - dev-test-tau
        - dev-train-synth
"""
import os
import glob
import shutil
import tqdm
from torchaudio import load, save
from torchaudio.functional import resample

# final dir with merged and sr-converted data
COMBINED_DIR = "/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/SELD_combined"


def convert_starss_sr(input_dir, output_dir, output_sr=32000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if "STARSS23" in input_dir:
        os.makedirs(os.path.join(output_dir, "foa_eval"))
        os.makedirs(os.path.join(output_dir, "foa_dev", "dev-train-sony"))
        os.makedirs(os.path.join(output_dir, "foa_dev", "dev-train-tau"))
        os.makedirs(os.path.join(output_dir, "foa_dev", "dev-test-sony"))
        os.makedirs(os.path.join(output_dir, "foa_dev", "dev-test-tau"))
    elif "SELD_synth" in input_dir:
        os.makedirs(os.path.join(output_dir, "foa_dev", "dev-train-synth"))

    for file in tqdm.tqdm(
        glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    ):
        data, sr = load(file)
        data = resample(data, orig_freq=sr, new_freq=output_sr)
        parent = os.path.basename(os.path.dirname(file))
        if "train" in parent or "test" in parent:  # For STARSS23's foa_dev
            save(
                src=data,
                uri=os.path.join(output_dir, "foa_dev", parent, os.path.basename(file)),
                sample_rate=output_sr,
            )
        else:  # For SELD_synth and STARSS23's foa_test
            save(
                src=data,
                uri=os.path.join(output_dir, parent, os.path.basename(file)),
                sample_rate=output_sr,
            )


if __name__ == "__main__":
    ### 1. Convert STARSS23 to 32000 Hz
    print("Converting STARSS23 train to 32000 Hz")
    convert_starss_sr(
        input_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/STARSS23/original",
        output_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/STARSS23/32000",
        output_sr=32000,
    )

    ### 2. Convert STARSS22 synthetic to 32000 Hz
    print("Converting STARSS22 synthetic to 32000 Hz")
    convert_starss_sr(
        input_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/SELD_synth/DCASE2022_SELD_synth_data/foa",
        output_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/SELD_synth/DCASE2022_SELD_synth_data/dev-train-synth",
        output_sr=32000,
    )

    ### 3. Copy wav files to COMBINED_DIR
    print(f"Copying STARSS23 wav files to {COMBINED_DIR}")
    os.makedirs(os.path.join(COMBINED_DIR, "foa_eval"))
    os.makedirs(os.path.join(COMBINED_DIR, "foa_dev", "dev-train-sony"))
    os.makedirs(os.path.join(COMBINED_DIR, "foa_dev", "dev-train-tau"))
    os.makedirs(os.path.join(COMBINED_DIR, "foa_dev", "dev-test-sony"))
    os.makedirs(os.path.join(COMBINED_DIR, "foa_dev", "dev-test-tau"))
    os.makedirs(os.path.join(COMBINED_DIR, "foa_dev", "dev-train-synth"))

    for file in tqdm.tqdm(
        glob.glob(os.path.join("/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/STARSS23/32000", "**", "*.wav"), recursive=True)
    ):
        parent = os.path.basename(os.path.dirname(file))
        if "train" in parent or "test" in parent: # For STARSS23's foa_dev
            shutil.copy(file, os.path.join(COMBINED_DIR, "foa_dev", parent, os.path.basename(file)))
        else: # For SELD_synth and STARSS23's foa_eval
            shutil.copy(file, os.path.join(COMBINED_DIR, "foa_eval", os.path.basename(file)))

    print(f"Copying STARSS22 synthetic wav files to {COMBINED_DIR}")
    for file in tqdm.tqdm(
        glob.glob(os.path.join("/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/SELD_synth/DCASE2022_SELD_synth_data/dev-train-synth", "*.wav"))
    ):
        parent = "dev-train-synth"
        shutil.copy(file, os.path.join(COMBINED_DIR, "foa_dev", parent, os.path.basename(file)))

    ### 4. Copy metadata to COMBINED_DIR
    print(f"Copying STARSS23 metadata to {COMBINED_DIR}")
    os.makedirs(os.path.join(COMBINED_DIR, "metadata_dev", "dev-train-sony"))
    os.makedirs(os.path.join(COMBINED_DIR, "metadata_dev", "dev-train-tau"))
    os.makedirs(os.path.join(COMBINED_DIR, "metadata_dev", "dev-test-sony"))
    os.makedirs(os.path.join(COMBINED_DIR, "metadata_dev", "dev-test-tau"))
    for file in tqdm.tqdm(
        glob.glob(
            os.path.join(
                "/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/STARSS23/original/metadata_dev",
                "**",
                "*.csv",
            )
        )
    ):
        parent = os.path.basename(os.path.dirname(file))
        shutil.copy(
            file,
            os.path.join(COMBINED_DIR, "metadata_dev", parent, os.path.basename(file)),
        )

    print(f"Copying STARSS22 synthetic metadata to {COMBINED_DIR}")
    os.makedirs(os.path.join(COMBINED_DIR, "metadata_dev", "dev-train-synth"))
    for file in tqdm.tqdm(
        glob.glob(
            os.path.join(
                "/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/SELD_synth/DCASE2022_SELD_synth_data/metadata",
                "*.csv",
            )
        )
    ):
        parent = "dev-train-synth"
        shutil.copy(
            file,
            os.path.join(COMBINED_DIR, "metadata_dev", parent, os.path.basename(file)),
        )
