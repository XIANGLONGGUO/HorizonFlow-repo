#!/usr/bin/env python3
"""Convert a training checkpoint (DCP directory) into a single bf16 .pt file for scripts/generate.py.

    python scripts/convert_checkpoint.py \
        outputs/rCM/HorizonFlow/horizonflow/checkpoints/iter_000000600/model horizonflow_600.pt

--key net_ema (default) exports the EMA weights, which are used for all results in the paper; --key net exports the
raw student weights.
"""
import argparse

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dcp_dir", help="<run>/checkpoints/iter_XXXXXXXXX/model")
    p.add_argument("out", help="output .pt file")
    p.add_argument("--key", default="net_ema", choices=["net_ema", "net"])
    a = p.parse_args()

    sd = {}
    _load_state_dict(sd, storage_reader=FileSystemReader(a.dcp_dir), planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    out = {
        k.replace(f"{a.key}.", "net.", 1): (v.to(torch.bfloat16) if v.is_floating_point() else v)
        for k, v in sd.items()
        if k.startswith(f"{a.key}.")
    }
    assert out, f"no '{a.key}.' keys in {a.dcp_dir}"
    torch.save(out, a.out)
    print(f"{len(out)} tensors ({a.key}) -> {a.out}")


if __name__ == "__main__":
    main()
