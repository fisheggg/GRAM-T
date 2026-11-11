# 26-10-17 Migration to Olivia

Code copied from fox `/projects/ec12/jinyueg/ambi_ssl` to Olivia `/cluster/projects/nn8104k/homes/jinyueg/ambi_ssl`.
Verified 730 entries (10,495,063 bytes of regular files): SHA-256 manifest digest `1ba2e35ac8e63d4c502a1941efa90fd857ad6326440f319af7f4cf13ae84f1b0` matched, including permissions and symlink targets.

Excluded checkpoint directories `GRAM-T/train_logs`, `GRAM-T/ckpt`, and `GRAM-T/seld-dcase-2022-GRAM/models`; checkpoint files matching `*.ckpt`, `*.safetensors`, `*.pth`, `*.pt`, and `*.h5`; Python caches; generated GRAM-T outputs, Slurm logs, smoke logs, SELD results and TensorBoard logs. Source code, Git history, uncommitted changes, configs and review reports are preserved.

Dataset source: fox `/cluster/work/projects/ec12/jinyueg/ambi_ssl/data`.
Dataset destination: Olivia `/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/data`.
Rclone copy runs directly on Olivia; source files are retained. Completion includes a full path/size check (no full dataset checksum scan). Workflow files and status are in `~/.cache/ambi_ssl_transfer`.

Training paths, environment, and Slurm scripts retain their fox settings and require adaptation before running on Olivia. The symlink `li_review/common/pyshim/dateutil` still points to the fox conda environment. Checkpoints are intentionally unavailable on Olivia.

Transfer runs in Olivia tmux session `ambi-ssl-transfer`. Fox was near its 500-thread user limit; rclone is limited to two SFTP connections with one transfer, one checker, and no per-file multi-thread copying. It enumerates/checks first, then copies. First-attempt log is retained as `rclone.attempt-1.log`.
