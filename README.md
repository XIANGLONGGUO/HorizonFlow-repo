# HorizonFlow: Few-Step Video Distillation in 600 Iterations

It contains the training and inference code of the main method:
distilling Wan2.1-T2V-1.3B into a 4-step video generator with plain DMD plus two corrections of the DMD target,
**Chunk Evenness (CE)** and **Semantic Commitment (SC)**, in 600 iterations (8.4k training samples).

The code is built on the open-source [rCM](https://github.com/NVlabs/rcm) distillation framework (Apache-2.0, see
`LICENSE.txt`). Everything specific to HorizonFlow is in the files listed below.

## Files

| File | What it does |
|---|---|
| `rcm/configs/experiments/rcm/horizonflow_wan2pt1_t2v.py` | The training configuration (`experiment=horizonflow_wan2pt1_1pt3B_t2v`): all hyperparameters of supplementary Tab. S1. |
| `rcm/models/t2v_model_distill_rcm.py` | Training loop. HorizonFlow adds the `ce_*` / `sc_*` options to `T2VDistillConfig_rCM` and, in `_student_dmd_step`, moves the DMD pseudo-target by `+delta_ce - delta_sc` (Eq. 9). |
| &nbsp;&nbsp;`_chunk_even_step`, `_ce_stats`, `_ce_window`, `_ce_init` | Chunk Evenness (Sec. 4.2): phase evenness E of a decoded latent window (Eq. 4), one-sided excess over the training clips (Eq. 5), trust-region steps (Eq. 6). |
| &nbsp;&nbsp;`_sc_term`, `_sc_coarse` | Semantic Commitment (Sec. 4.3): guided teacher prediction from the student's last input (Eq. 7), coarse time-averaged difference with gate and cap (Eq. 8). |
| `scripts/train_horizonflow.sh` | Launches training with `torchrun`. |
| `scripts/convert_checkpoint.py` | Converts a training checkpoint (DCP) into a single `.pt` for inference. |
| `scripts/generate.py` | 4-step text-to-video sampling (832x480, 81 frames), single prompt or a prompt list sharded over GPUs. |
| `scripts/train.py`, `imaginaire/`, `rcm/` (rest) | The rCM training framework (FSDP2, Wan2.1 networks, VAE, umT5, data loading), unchanged except for small compatibility fixes. |

The critic update, the sampler, and the network are those of plain DMD; the distilled model generates exactly like
a standard 4-step student.

## Environment

Tested with Python 3.12, PyTorch 2.7.1 (CUDA 11.8), FlashAttention 2.7.4.post1, on A100-80GB GPUs.

```bash
conda create -n horizonflow python=3.12 && conda activate horizonflow
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install megatron-core hydra-core loguru attrs fvcore nvidia-ml-py "imageio[ffmpeg]" pandas wandb psutil \
            ftfy regex transformers webdataset safetensors einops av
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`transformer_engine` is not needed: the configuration uses `torch.optim.AdamW`.

## Data and checkpoints

HorizonFlow uses the same teacher files and training data as rCM. Run the commands from the repository root:

```bash
# Wan2.1-T2V-1.3B teacher, Wan2.1 VAE, umT5 text encoder, and the umT5 embedding of Wan's negative prompt
git clone https://huggingface.co/worstcoder/Wan assets/checkpoints
python -m torch.distributed.checkpoint.format_utils torch_to_dcp \
    assets/checkpoints/Wan2.1-T2V-1.3B.pth assets/checkpoints/Wan2.1-T2V-1.3B.dcp

# Training data synthesized by Wan2.1-14B (webdataset shards of video latents, text embeddings, and captions)
git clone https://huggingface.co/datasets/worstcoder/Wan_datasets assets/datasets
```

Expected layout (other locations can be passed with `CKPT=...` and `DATA=...`):

```
assets/checkpoints/Wan2.1-T2V-1.3B.dcp/           teacher, also initializes the student and the critic
assets/checkpoints/Wan2.1_VAE.pth
assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth
assets/checkpoints/umT5_wan_negative_emb.pt
assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar
```

600 iterations draw only 8.4k samples, so any subset of the shards works; the paper's run read from all of them.

## Easy start

### 1. Train

```bash
GPUS=0,1,2,3,4,5,6 bash scripts/train_horizonflow.sh
```

- The global batch is the number of GPUs times 2 (gradient accumulation). The paper uses 7 GPUs, i.e. 14 videos per
  iteration and 8.4k samples over 600 iterations (300 student and 300 critic updates).
- On 7 A100-80GB GPUs an iteration takes about 44 s, so the run takes about 7.3 hours. The first iteration is slower
  because kernels are compiled.
- Checkpoints are written every 150 iterations to
  `outputs/rCM/HorizonFlow/horizonflow/checkpoints/iter_000000{150,300,450,600}/`.
- Any configuration value can be overridden on the command line, e.g. `trainer.max_iter=300`.

### 2. Convert the checkpoint

```bash
python scripts/convert_checkpoint.py \
    outputs/rCM/HorizonFlow/horizonflow/checkpoints/iter_000000600/model horizonflow_600.pt
```

The EMA weights (`--key net_ema`, the default) are used for all results in the paper.

### 3. Generate videos

```bash
PYTHONPATH=. python scripts/generate.py --ckpt horizonflow_600.pt --out-dir samples \
    --prompt "A person is washing dishes."
```

One video takes about 11 s on an A100 (4 network evaluations). Useful options:

- `--prompts-file prompts.txt` (one prompt per line) and `--shard i --nshards n` to split a list over GPUs;
- `--samples 5 --seed 42` for five videos per prompt with seeds 42-46, saved as `<name>-<i>.mp4`;
- `--names-file` to name the files differently from the conditioning text. For VBench, pass VBench's augmented
  prompts as `--prompts-file` and the original prompts, in the same order, as `--names-file`;
- `--vae`, `--text-encoder`, `--tokenizer` if the Wan2.1 files are not in `assets/checkpoints` (the umT5 tokenizer
  is downloaded from Hugging Face unless a local directory is given);
- `--steps 3` for 3-step sampling.

## License

Apache License 2.0 (`LICENSE.txt`), inherited from rCM. The SPDX headers of the rCM files are kept.

