"""Pack the Wham data into WebDataset .tar files,
as the sharded version is not provided by GRAM-T authors"""

import os
import glob
import shutil
import webdataset as wds
import pathlib
import torchaudio
import tqdm
from torchaudio.transforms import Resample


def pack_wham_data(data_dir, output_dir, sr=32000, maxcount=1000):
    """Pack the Wham data into WebDataset .tar files"""
    data_dir = pathlib.Path(data_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(data_dir.glob("*.wav"))
    print(f"=> Found {len(files)} .wav files in {data_dir}")

    # init resampler
    _, sr_ = torchaudio.load(files[0])
    resampler = None
    if sr_ != sr:
        resampler = Resample(sr_, sr)

    idx = 0
    with wds.ShardWriter(f"{output_dir}/shard-%06d.tar", maxcount=maxcount) as writer:
        for file in tqdm.tqdm(files):
            audio, sr_ = torchaudio.load(file)
            if resampler:
                audio = resampler(audio)
            example = {
                "__key__": f"{idx:06}",
                "npy": audio.numpy(),
            }
            writer.write(example)
            idx += 1


if __name__ == "__main__":
    data_dir = "/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/wham_noise/tr/"
    output_dir = (
        "/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/wham_noise_sharded/tr/"
    )

    pack_wham_data(data_dir, output_dir)
