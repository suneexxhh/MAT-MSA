# MAT+MSA

This repository contains the MAT+MSA ablation used in the MSCM paper. It
extends the Multi-Agent Transformer (MAT) encoder with multi-scale attention.
It does not include the Titans long-term memory or the role-aware fusion
module used by the complete MSCM model.

The implementation keeps MAT's autoregressive decoder and PPO training
procedure. Its encoder produces scale-specific interaction representations and
projects their concatenation back to the encoder dimension.

## Repository Layout

```text
mat/
  algorithms/mat/algorithm/       Encoder, decoder, and scale configuration
  envs/starcraft2/                SMAC environment wrapper and map registry
  runner/                         PPO training and evaluation runners
  scripts/train/train_smac.py     SMAC training entry point
  scripts/train_smac_*_custom.sh  Map-specific training scripts
install_sc2.sh                    StarCraft II and SMAC map installer
requirements.txt                  Python dependencies
```

Training outputs are intentionally excluded from this repository. New
checkpoints and TensorBoard summaries are written below
`mat/scripts/results/`, and custom-script logs are written below
`mat/scripts/logs/`.

## Installation

The original experiments use Linux, Python 3.8 or 3.9, PyTorch 1.10, and a
CUDA-capable GPU. Create and activate an isolated environment first:

```bash
conda create -n mat-msa python=3.8 -y
conda activate mat-msa
```

Install the dependencies from the repository root:

```bash
cd /path/to/MAT+MSA
pip install -r requirements.txt
```

Install StarCraft II and the SMAC maps:

```bash
bash install_sc2.sh
export SC2PATH="$(pwd)/3rdparty/StarCraftII"
```

The installer places the game and maps under `3rdparty/`. If StarCraft II is
already installed elsewhere, set `SC2PATH` to that installation and verify
that `Maps/SMAC_Maps/` contains the SMAC maps. The installer downloads the
legacy SMAC package used by this codebase.

## Table 1 Configuration

Automatic configuration is enabled by default. The map name selects the
number of scales and the first nontrivial chunk size `c1` according to Table 1
of the paper. The resulting hierarchy is `[1, c1, N]` for three scales and
`[1, c1, c1^2, N]` for four scales.

| SMAC map | Scales | `c1` | Attention hierarchy |
|---|---:|---:|---|
| `1c3s5z` | 3 | 3 | `[1, 3, 9]` |
| `3s5z` | 3 | 4 | `[1, 4, 8]` |
| `5m_vs_6m` | 3 | 2 | `[1, 2, 5]` |
| `8m_vs_9m` | 3 | 4 | `[1, 4, 8]` |
| `10m_vs_11m` | 3 | 4 | `[1, 4, 10]` |
| `6h_vs_8z` | 3 | 4 | `[1, 4, 6]` |
| `3s5z_vs_3s6z` | 3 | 3 | `[1, 3, 8]` |
| `MMM2` | 3 | 3 | `[1, 3, 10]` |
| `27m_vs_30m` | 4 | 3 | `[1, 3, 9, 27]` |

The mapping is defined once in
`mat/algorithms/mat/algorithm/scale_config.py` and is used by both the
training entry point and the encoder. Maps outside this table use the
command-line defaults of three scales and `c1=4`.

## Training

Run commands from `mat/scripts`, because the supplied scripts use relative
paths to the training entry point.

### One map and one seed

Each custom script takes a GPU ID and optionally a seed. Automatic Table 1
configuration is used when only these arguments are supplied:

```bash
cd /path/to/MAT+MSA/mat/scripts
bash train_smac_27m_vs_30m_custom.sh 0 1
```

Here `0` selects the first visible GPU and `1` is the random seed. The command
automatically uses four scales with hierarchy `[1, 3, 9, 27]`.

The script prints the resolved configuration at startup and saves a timestamped
log under `mat/scripts/logs/`. Checkpoints and TensorBoard summaries are saved
under `mat/scripts/results/StarCraft2/<map>/mat/<experiment>/`.

### Manual scale override

To bypass the Table 1 mapping, pass `n_levels` and `c1` after the GPU ID. The
script then disables map adaptation internally:

```bash
bash train_smac_27m_vs_30m_custom.sh 0 3 4 1
```

This runs seed `1` with three scales and `c1=4`, producing `[1, 4, 27]`.
The general argument order is:

```text
bash train_smac_<map>_custom.sh <gpu_id> [<seed>]
bash train_smac_<map>_custom.sh <gpu_id> <n_levels> <c1> [<seed>]
```

For example, the Table 1 configuration for `MMM2` can be launched with:

```bash
bash train_smac_MMM2_custom.sh 0 1
```

### Multiple seeds

The following loop launches five seeds sequentially on GPU 0:

```bash
for seed in 1 2 3 4 5; do
  bash train_smac_6h_vs_8z_custom.sh 0 "$seed"
  sleep 60
done
```

To distribute jobs across GPUs, start one command per GPU or adapt
`run_all.sh`. Before using a batch launcher, check its map list and GPU list
against the resources available on your machine.

## Evaluation and Monitoring

The custom scripts include `--use_eval`, so evaluation runs during training.
Monitor a running job with:

```bash
tail -f logs/mat_msa_27m_vs_30m_adaptive_seed1_*.log
```

The exact log filename is printed when the job starts. TensorBoard summaries
can be inspected with:

```bash
tensorboard --logdir results/StarCraft2
```

Use `nvidia-smi` to verify GPU allocation. To stop a job, identify its process
with `ps` and terminate that process normally.

## Troubleshooting

- **SC2 cannot be found:** export `SC2PATH` to the directory containing the
  `StarCraftII` executable and verify `Maps/SMAC_Maps/`.
- **Import errors:** run commands from `mat/scripts`, or set
  `PYTHONPATH=/path/to/MAT+MSA`.
- **CUDA errors:** verify the PyTorch build with
  `python -c "import torch; print(torch.cuda.is_available())"` and use a valid
  GPU ID.
- **Unknown map:** use one of the map names listed in the Table 1 section and
  check that the corresponding SMAC map is installed.
- **Reproducibility:** keep the seed, training budget, rollout settings, and
  PPO settings fixed when comparing runs.

## Citation

If this code is useful in your research, please cite the MAT paper and the
MSCM paper associated with this implementation.

```bibtex
@article{wen2022multi,
  title={Multi-Agent Reinforcement Learning is a Sequence Modeling Problem},
  author={Wen, Muning and Kuba, Jakub Grudzien and Lin, Runji and Zhang, Weinan and Wen, Ying and Wang, Jun and Yang, Yaodong},
  journal={arXiv preprint arXiv:2205.14953},
  year={2022}
}
```
