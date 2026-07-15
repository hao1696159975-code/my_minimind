#!/usr/bin/env python3
"""Count MiniMind parameters (Dense / MoE).

Usage (from repo root, after upstream is cloned):

  python scripts/lab/count_params.py
  python scripts/lab/count_params.py --upstream upstream --moe
  python scripts/lab/count_params.py --hidden 768 --layers 8

Does not require pretrained weights. Fails clearly if upstream is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_upstream(upstream: Path) -> None:
    if not upstream.exists():
        raise SystemExit(
            f"[error] upstream not found: {upstream.resolve()}\n"
            "Clone first:\n"
            "  git submodule add https://github.com/jingyaogong/minimind.git upstream\n"
            "or:\n"
            "  git clone --depth 1 https://github.com/jingyaogong/minimind.git upstream"
        )
    sys.path.insert(0, str(upstream.resolve()))


def main() -> None:
    p = argparse.ArgumentParser(description="Count MiniMind params")
    p.add_argument("--upstream", type=Path, default=Path("upstream"))
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--moe", action="store_true", help="Enable MoE config if supported")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _add_upstream(args.upstream)

    import torch

    # Import surface may evolve; keep diagnostics explicit.
    try:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    except Exception as e:  # noqa: BLE001 - teaching script, show real import errors
        raise SystemExit(
            f"[error] failed to import MiniMind from {args.upstream}: {e}\n"
            "Open upstream/model/model_minimind.py and adjust this script's import."
        ) from e

    torch.manual_seed(args.seed)

    cfg_kwargs = {
        "hidden_size": args.hidden,
        "num_hidden_layers": args.layers,
    }

    # Best-effort MoE toggles across versions
    if args.moe:
        for key, val in (
            ("use_moe", True),
            ("moe", True),
            ("n_routed_experts", 4),
            ("num_experts_per_tok", 1),
        ):
            cfg_kwargs[key] = val

    try:
        config = MiniMindConfig(**cfg_kwargs)
    except TypeError:
        # Fall back to defaults then setattr
        config = MiniMindConfig()
        for k, v in cfg_kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif args.moe and k in {"use_moe", "moe"}:
                print(f"[warn] config has no field {k}; MoE may need manual flag")

    model = MiniMindForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== MiniMind param count ===")
    print(f"upstream:   {args.upstream.resolve()}")
    print(f"config:     {config}")
    print(f"total:      {total:,}  ({total/1e6:.3f} M)")
    print(f"trainable:  {trainable:,}  ({trainable/1e6:.3f} M)")
    if args.moe:
        print(
            "note: MoE *active* params ≈ backbone + top-k experts, not total.\n"
            "      Compute active from router top-k after reading model_minimind.py."
        )


if __name__ == "__main__":
    main()
