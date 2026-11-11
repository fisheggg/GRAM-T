# 26-10-17 Conda environments moved/prepared on Olivia

Copied the requested x86 gram-t prefix into `/cluster/projects/nn8104k/homes/jinyueg/conda/envs/gram-t`. Created the conda/envs folders. Relocated prefixes with conda-pack and verified the archive SHA-256. Existing conda/pip conflicts required restoring source-installed pip/setuptools/pkg_resources/wheel files instead of their stale conda-cache versions. Verified all 15,796 Python files after prefix normalization, 185 distribution records, Python/NumPy/SSL, pip and activation. Source environment remains intact. External Fox user-site libraries are separately inventoried, not included in the specified-prefix snapshot.

Prepared `/cluster/projects/nn8104k/homes/jinyueg/conda/envs/gram-t-gpu`, an ARM64 container-backed Conda environment following Olivia's software policy. Python 3.11.17, PyTorch 2.7.1+cu128, CUDA 12.8, NumPy 2.0.2 and Lightning 2.4.0. Build job 2562031 and runtime validation job 2562048 completed successfully on GH200 GPUs. CUDA gradients, audio transforms, project model/datamodule imports, a Lightning GPU step and TensorBoard logging passed. No training checkpoints were created.

Activation: inside an ARM64 GPU allocation, `source /cluster/projects/nn8104k/homes/jinyueg/conda/activate-gram-t-gpu.sh`.
Details and reproducible inputs: `/cluster/projects/nn8104k/homes/jinyueg/conda/README.md`, `/cluster/projects/nn8104k/homes/jinyueg/conda/build/`.

The dataset rclone copy continues in tmux `ambi-ssl-transfer`. No original files were removed from Fox and no training configurations or model code were changed.
