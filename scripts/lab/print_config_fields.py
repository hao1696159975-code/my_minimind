#!/usr/bin/env python3
"""Print MiniMindConfig fields from upstream (no GPU required if torch+transformers installed).

  python scripts/lab/print_config_fields.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    upstream = root / "upstream"
    target = upstream / "model" / "model_minimind.py"
    if not target.is_file():
        raise SystemExit(
            f"missing {target}\n"
            "Run: git submodule update --init --recursive"
        )
    sys.path.insert(0, str(upstream))
    from model.model_minimind import MiniMindConfig

    dense = MiniMindConfig(use_moe=False)
    moe = MiniMindConfig(use_moe=True)
    keys = sorted(k for k in vars(dense) if not k.startswith("_"))
    print("MiniMindConfig fields (Dense defaults):")
    for k in keys:
        print(f"  {k} = {getattr(dense, k)!r}")
    print("\nMoE-related when use_moe=True:")
    for k in (
        "use_moe",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "norm_topk_prob",
        "router_aux_loss_coef",
    ):
        print(f"  {k} = {getattr(moe, k)!r}")


if __name__ == "__main__":
    main()
