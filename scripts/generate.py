#!/usr/bin/env python3
"""4-step text-to-video generation with a distilled HorizonFlow checkpoint (Wan2.1-T2V-1.3B, 832x480, 81 frames).

Single prompt:
    PYTHONPATH=. python scripts/generate.py --ckpt horizonflow_600.pt --out-dir samples \
        --prompt "A person is washing dishes."

Many prompts, sharded over GPUs (one process per GPU):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/generate.py --ckpt horizonflow_600.pt --out-dir samples \
        --prompts-file prompts.txt --shard 0 --nshards 8

Video i of a prompt uses seed (--seed + i) and is saved as "<name>-<i>.mp4", the file naming of VBench.  For VBench,
pass the augmented prompts as --prompts-file and the original prompts (same order) as --names-file; the paper uses
--samples 5 (seeds 42-46).

Sampler: the rCM / TrigFlow multistep sampler with sigma_max 1600 and intermediate TrigFlow times [1.5, 1.4, 1.0];
each step predicts x0 and re-noises it to the next time with fresh noise.
"""
import argparse
import math
import time
from pathlib import Path

import av
import numpy as np
import torch

from imaginaire.lazy_config import instantiate
from rcm.inference.wan2pt1_t2v_rcm_infer import WAN2PT1_1PT3B_T2V
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.model_utils import init_weights_on_device, load_state_dict
from rcm.utils.umt5 import UMT5EncoderModel


def to_uint8(video: torch.Tensor) -> np.ndarray:
    """[C, T, H, W] in [-1, 1] -> [T, H, W, C] uint8."""
    video = ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return video.permute(1, 2, 3, 0).cpu().numpy()


def write_mp4(frames: np.ndarray, path: Path, fps: int = 16, lossless: bool = False) -> None:
    tmp = path.with_suffix(".part.mp4")
    with av.open(str(tmp), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"qp": "0", "bf": "0"} if lossless else {"crf": "18"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    tmp.rename(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help=".pt from scripts/convert_checkpoint.py")
    p.add_argument("--out-dir", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="a single text prompt")
    g.add_argument("--prompts-file", type=Path, help="one prompt per line")
    p.add_argument("--names-file", type=Path, default=None, help="output names, one per line (default: the prompts)")
    p.add_argument("--samples", type=int, default=1, help="videos per prompt")
    p.add_argument("--seed", type=int, default=42, help="video i of a prompt uses seed + i")
    p.add_argument("--steps", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--sigma-max", type=float, default=1600.0)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--vae", default="assets/checkpoints/Wan2.1_VAE.pth")
    p.add_argument("--text-encoder", default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--tokenizer", default="google/umt5-xxl", help="Hugging Face name or local directory")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--lossless", action="store_true", help="lossless H.264 (qp 0) instead of crf 18")
    a = p.parse_args()
    dev = torch.device("cuda")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [a.prompt] if a.prompt else [l.rstrip("\n") for l in open(a.prompts_file) if l.strip()]
    names = [l.rstrip("\n") for l in open(a.names_file) if l.strip()] if a.names_file else prompts
    assert len(names) == len(prompts), "--names-file must have one line per prompt"
    jobs = list(dict.fromkeys(zip(names, prompts)))
    todo = [
        (n, t)
        for j, (n, t) in enumerate(jobs)
        if j % a.nshards == a.shard and not all((a.out_dir / f"{n}-{i}.mp4").is_file() for i in range(a.samples))
    ]
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} prompts x {a.samples} videos", flush=True)
    if not todo:
        return

    enc = UMT5EncoderModel(text_len=512, dtype=torch.bfloat16, device=dev, checkpoint_path=a.text_encoder, tokenizer_path=a.tokenizer)
    with init_weights_on_device():
        net = instantiate(WAN2PT1_1PT3B_T2V).eval()
    state = load_state_dict(a.ckpt)
    state = {k[len("net.") :] if k.startswith("net.") else k: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False, assign=True)
    assert not unexpected and all("accum_" in k for k in missing), (missing, unexpected)
    net = net.to(dev, torch.bfloat16)
    vae = Wan2pt1VAEInterface(vae_pth=a.vae, chunk_duration=a.frames)

    shape = (1, 16, 1 + (a.frames - 1) // 4, 60, 104)  # latent of 832 x 480 x frames
    mid = [1.5, 1.4, 1.0][: a.steps - 1]
    tt = torch.tensor([math.atan(a.sigma_max), *mid, 0.0], dtype=torch.float64, device=dev)
    tt = torch.sin(tt) / (torch.cos(tt) + torch.sin(tt))  # TrigFlow -> rectified-flow time
    ones = torch.ones(1, 1, device=dev, dtype=torch.float64)

    started, done = time.time(), 0
    for n_done, (name, text) in enumerate(todo, 1):
        with torch.no_grad():
            emb = enc([text], device=dev).to(torch.bfloat16)
        for i in range(a.samples):
            out = a.out_dir / f"{name}-{i}.mp4"
            if out.is_file():
                continue
            gen = torch.Generator(device=dev).manual_seed(a.seed + i)
            x = torch.randn(shape, dtype=torch.float32, device=dev, generator=gen).double() * tt[0]
            with torch.no_grad():
                for t_cur, t_next in zip(tt[:-1], tt[1:]):
                    v = net(x_B_C_T_H_W=x.to(torch.bfloat16), timesteps_B_T=(t_cur * ones * 1000).to(torch.bfloat16), crossattn_emb=emb).double()
                    x0 = x - t_cur * v
                    x = (1 - t_next) * x0 + t_next * torch.randn(shape, dtype=torch.float32, device=dev, generator=gen).double()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video = vae.decode(x.float())[0]
            write_mp4(to_uint8(video), out, lossless=a.lossless)
            done += 1
        print(f"prompts {n_done}/{len(todo)}, videos {done}, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
