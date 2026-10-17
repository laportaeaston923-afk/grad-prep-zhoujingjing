#!/usr/bin/env python3
"""Copy real upstream metrics/plots to this experiment's results folder."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / ".upstream" / "official" / "outputs" / "maze9_metrics"
RESULTS = ROOT / "results"


def export_metrics(source: Path, target: Path) -> list[str]:
    if not source.is_dir():
        raise FileNotFoundError(f"没有找到上游结果目录：{source}。请先完成 eval 阶段。")
    patterns = [
        "maze9_vae_vs_grwm_norm_mse_per_step.*",
        "*_rollout_pixel_mse_summary.json",
        "maze9_rollout_pixel_mse_table.json",
    ]
    candidates = {p for glob in patterns for p in source.glob(glob) if p.is_file()}
    if not candidates:
        raise FileNotFoundError("未找到真实的 VAE/GRWM MSE 文件；不会生成虚构结果图。")
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(candidates):
        shutil.copy2(path, target / path.name)
        copied.append(path.name)
    return copied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=METRICS)
    p.add_argument("--target", type=Path, default=RESULTS)
    args = p.parse_args()
    try:
        files = export_metrics(args.source, args.target)
    except (FileNotFoundError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print("已导出真实计算文件：", ", ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
