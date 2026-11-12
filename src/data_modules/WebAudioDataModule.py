import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import webdataset as wds
from .dataset_functions import pre_process_audio, pre_process_noise
from .scene_module import generate_scenes
import torch
import multiprocessing as mp
import queue
import threading
import time
import traceback


def to_torch(sample):
    return torch.from_numpy(sample[0])


def _report_loader_error(name, error):
    """Log a background loader failure and keep it for consumers."""
    tb = traceback.format_exc()
    print(f"[{name}] Loader failed:\n{tb}", flush=True)
    error.append(tb)


# If a loader produces nothing for this long, something is wrong (dead or
# deadlocked loader process). Fail loudly instead of polling forever.
LOADER_STALL_TIMEOUT_S = 900.0


def _get_or_retry(data_queue, error, timeout, name):
    """Block until a sample arrives without growing the Python stack."""
    t0 = time.monotonic()
    while True:
        try:
            return data_queue.get(timeout=timeout)
        except queue.Empty:
            if len(error) > 0:
                raise RuntimeError(f"{name} loader failed:\n{error[0]}") from None
            waited = time.monotonic() - t0
            if waited > LOADER_STALL_TIMEOUT_S:
                raise RuntimeError(
                    f"{name} loader produced no sample for {waited:.0f}s. "
                    "The loader process is probably dead or deadlocked."
                )


def _watch_parent(parent_pid):
    """Exit if the training process dies, so loader processes are not left behind."""
    while True:
        if os.getppid() != parent_pid:
            os._exit(0)
        time.sleep(1.0)


def _init_loader_process():
    """Common setup for the forked noise/RIR loader processes.

    These processes are forked after the main process has already run OpenMP
    parallel regions (model init). libgomp's thread pool does not survive
    fork, so the first parallel CPU op in the child (e.g. the float64->float32
    copy in `.float()`, or the copy into shared memory when a tensor is put
    on the queue) blocks forever on a barrier. torch's own DataLoader workers
    avoid this by running single-threaded; do the same here.
    """
    torch.set_num_threads(1)
    threading.Thread(
        target=_watch_parent, args=(os.getppid(),), daemon=True
    ).start()


def _iter_npy_tensors(url, shuffle_buffer=100):
    """Read npy shards in this process.

    A nested DataLoader copies the shuffle buffer into every worker. Each RIR
    sample is about 18MB, so those workers were large enough to be OOM-killed.
    """

    def to_torch(sample):
        return torch.from_numpy(sample[0]).float()

    dataset = (
        wds.WebDataset(url, resampled=True, shardshuffle=True)
        .repeat()
        .shuffle(shuffle_buffer)
        .decode("pil")
        .to_tuple("npy")
        .map(to_torch)
    )
    return iter(dataset)


class NoiseDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""

    def __init__(
        self, noise_data_dir: str, buffer_size: int = 500, num_workers: int = 1
    ):
        self.noise_data_dir = noise_data_dir
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.noise_queue = self.manager.Queue(maxsize=buffer_size)
        self.stop_event = self.manager.Event()
        self.error = self.manager.list()
        self.processes = []
        self.started = False

    def _worker(self):
        """Worker process to load noise data."""
        _init_loader_process()
        samples = _iter_npy_tensors(self.noise_data_dir)
        print(f"[NOISE] Loader initialized at {time.strftime('%H:%M:%S')}", flush=True)

        while not self.stop_event.is_set():
            try:
                noise = next(samples)
                self.noise_queue.put(noise, timeout=1.0)
            except queue.Full:
                continue
            except Exception:
                _report_loader_error("NOISE", self.error)
                raise

    def start(self):
        """Start the Noise loading process."""
        if not self.started:
            self.process = mp.Process(target=self._worker, daemon=True)
            self.process.start()
            self.started = True
            print(
                f"[NOISE] Manager started with buffer size: {self.buffer_size}",
                flush=True,
            )
        return self

    def __next__(self, timeout: float = 1.0):
        """Get Noise data from the queue."""
        return _get_or_retry(self.noise_queue, self.error, timeout, "NOISE")

    def stop(self):
        """Stop the Noise loading process."""
        if self.started:
            self.stop_event.set()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
            self.started = False

    def __getstate__(self):
        # DataLoader workers started with spawn/forkserver get only the queue/event/error proxies,
        # not the manager or loader process; their copy never stops the shared loader.
        state = self.__dict__.copy()
        for key in ("manager", "process", "processes"):
            state.pop(key, None)
        state["started"] = False
        return state

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()


class RIRDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""

    def __init__(self, rir_data_dir: str, buffer_size: int = 128, num_workers: int = 0):
        self.rir_data_dir = rir_data_dir
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.rir_queue = self.manager.Queue(maxsize=buffer_size)
        self.stop_event = self.manager.Event()
        self.error = self.manager.list()
        self.processes = []
        self.started = False

    def _worker(self):
        """Worker process to load RIR data."""
        _init_loader_process()
        samples = _iter_npy_tensors(self.rir_data_dir)
        print(f"[RIR] Loader initialized at {time.strftime('%H:%M:%S')}", flush=True)

        while not self.stop_event.is_set():
            try:
                rirs = next(samples)
                self.rir_queue.put(rirs, timeout=1.0)
            except queue.Full:
                continue
            except Exception:
                _report_loader_error("RIR", self.error)
                raise

    def start(self):
        """Start the RIR loading process."""
        if not self.started:
            self.process = mp.Process(target=self._worker, daemon=True)
            self.process.start()
            self.started = True
        return self

    def __next__(self, timeout: float = 1.0):
        """Get RIR data from the queue."""
        return _get_or_retry(self.rir_queue, self.error, timeout, "RIR")

    def stop(self):
        """Stop the RIR loading process."""
        if self.started:
            self.stop_event.set()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
            self.started = False

    def __getstate__(self):
        # DataLoader workers started with spawn/forkserver get only the queue/event/error proxies,
        # not the manager or loader process; their copy never stops the shared loader.
        state = self.__dict__.copy()
        for key in ("manager", "process", "processes"):
            state.pop(key, None)
        state["started"] = False
        return state

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()


class WebAudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str,
        val_data_dir: str,
        rir_data_dir: str,
        base_noise_dir: str,
        batch_size: int = 32,
        sr: int = 32000,
        masker=None,
        nr_samples_per_audio: int = 16,
        nr_patches: int = 200,
        cache_size: int = 1000,
        num_workers: int = 16,
        mp_context: str = None,
        with_noise: bool = True,
        with_rir: bool = True,
        # rir_duration: int = 160000, # 32000 * 5
        **kwargs,
    ):
        """Initialize the data module with shared noise data."""
        super().__init__()
        self.datapath = base_data_dir
        self.val_path = val_data_dir
        self.noise_dir = base_noise_dir
        self.batch_size = batch_size
        self.sr = sr
        self.masker = masker
        self.nr_samples_per_audio = nr_samples_per_audio
        self.nr_patches = nr_patches
        self.cache_size = cache_size
        self.num_workers = num_workers
        # Start method of the DataLoader workers: None = fork (the default), or "forkserver"/"spawn".
        self.mp_context = mp_context

        self.with_noise = with_noise
        self.with_rir = with_rir
        # self.rir_duration = rir_duration

        if self.with_noise:
            self.noise_loader = NoiseDataManager(base_noise_dir).start()

        if self.with_rir:
            self.rir_loader = RIRDataManager(rir_data_dir).start()

    def __getstate__(self):
        # Workers started with spawn/forkserver receive this module through the bound _augment_sample;
        # leave out the Lightning trainer it is attached to.
        state = self.__dict__.copy()
        for key in ("trainer", "_trainer"):
            state.pop(key, None)
        return state

    def _augment_sample(self, sample):
        """Augment sample with noise and RIR data."""
        # print("=> start augmenting sample")

        audio, audio_sr = sample[0]
        audio = pre_process_audio(audio, audio_sr, self.sr)
        # Initialize all variables
        noise = None
        noise_rirs = None
        snr = None
        source_rir = None
        noise_start_idx = 0
        noise_length = 0

        if self.with_rir:
            # print("=> start loading rirs")
            rirs = next(self.rir_loader)
            source_rir = rirs[0]
            # print(f"=> rirs shape: {rirs.shape}")
            # fix the length of rirs
            # rirs = pad_or_truncate_1d(rirs, self.rir_duration)
            # source_rir = rirs[0].unsqueeze(0)

        if self.with_noise:
            if self.with_rir:
                noise_rirs = rirs[1:]

            noise = next(self.noise_loader)
            # print("=> rirs and noise loaded")
            noise = pre_process_noise(noise, audio_sr=32000, resample_sr=self.sr)
            # This function already handles the randomly cropping noise to match audio length
            # if noise is bigger than the audio length.
            noise = generate_scenes.fade_noise(noise, audio, self.sr)
            noise_length = noise.shape[-1]
            # If audio is bigger than noise, then we will place the noise in a random location of the audio
            if audio.shape[-1] > noise.shape[-1]:
                noise_start_idx = torch.randint(
                    0, audio.shape[-1] - noise.shape[-1], (1,)
                ).item()
                new_agg_noise = torch.zeros_like(audio)
                new_agg_noise[noise_start_idx : noise_start_idx + noise.shape[-1]] = (
                    noise
                )
                noise = new_agg_noise
            snr = torch.distributions.uniform.Uniform(5, 40).sample().item()

        context_idx = self.masker(
            local_features=None,
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_patches,
        )
        # print(f"=> audio shape: {audio.shape}")
        # print(f"=> noise shape: {noise.shape}")
        # print(f"=> noise length: {noise_length}")
        # print(f"=> noise_start_idx: {noise_start_idx}")
        # print(f"=> source_rir shape: {source_rir.shape}")
        # print(f"=> noise_rirs shape: {noise_rirs.shape}")
        # print(f"=> snr: {snr}")
        # print(f"=> context_idx: {context_idx}")
        return (
            audio,
            noise,
            noise_length,
            noise_start_idx,
            source_rir,
            noise_rirs,
            snr,
            context_idx,
        )

    def make_web_dataset(
        self, path: str, split_scene: str, split_noise: str, shuffle: int
    ):
        """Create a WebDataset pipeline for audio processing."""
        dataset = (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=False,
            )
            .repeat()
            .shuffle(shuffle)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(self._augment_sample)
            .batched(self.batch_size)
        )
        return dataset

    def setup(self, stage: str):
        """Set up datasets for training."""
        if stage == "fit":
            self.audio_train = self.make_web_dataset(
                self.datapath, "train", "tr", shuffle=1000
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context=self.mp_context if self.num_workers > 0 else None,
        )
        return loader
