"""Print the exact Stage-2 files imported by the active Python environment."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import common
import flow_transport
import latent_stats
import pano_dit
import pano_rae
import sample_pano_dit
import train_pano_dit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="")
    args = p.parse_args()

    modules = [
        common, flow_transport, latent_stats, pano_dit,
        pano_rae, train_pano_dit, sample_pano_dit,
    ]
    for module in modules:
        print(f"{module.__name__}: {Path(inspect.getfile(module)).resolve()}")
    print("save_panorama_batch signature:", inspect.signature(common.save_panorama_batch))
    print("objective:", train_pano_dit.OBJECTIVE)

    if args.output_dir:
        root = Path(args.output_dir)
        files = sorted((p for p in root.rglob("*.png") if p.is_file()),
                       key=lambda p: p.stat().st_mtime)
        print("\nNewest PNG files:")
        for path in files[-30:]:
            print(path.resolve())
        tangent = [p for p in files if "tangent_grid" in p.name]
        print(f"\nTangent-grid files under output_dir: {len(tangent)}")
        if tangent:
            print("These are stale/diagnostic files; v4 normal sampling does not create them.")


if __name__ == "__main__":
    main()
