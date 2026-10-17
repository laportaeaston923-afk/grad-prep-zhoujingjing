#!/usr/bin/env python3
"""Reproducible entrypoint around the unmodified official GRWM model code.

Only the upstream *launcher* is patched in a temporary copy so AE jobs run
sequentially on one GPU. No network architecture, training loss or sampler is
reimplemented here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "upstream.lock.json"
UPSTREAM = ROOT / ".upstream" / "official"
STAGES = ("ae", "precompute", "dynamics", "eval", "all")


def read_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], **kwargs) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def ensure_source(download: bool) -> None:
    lock = read_lock()
    if not (UPSTREAM / ".git").exists():
        if not download:
            raise RuntimeError("未下载官方代码：请先运行 python code/run.py bootstrap")
        UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", lock["repo"], str(UPSTREAM)])
        run_cmd(["git", "-C", str(UPSTREAM), "checkout", "--detach", lock["commit"]])
    result = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if result != lock["commit"]:
        raise RuntimeError(f"官方代码版本不一致：本地 {result}，锁定 {lock['commit']}")
    if not (UPSTREAM / "run_maze9_vae_grwm.sh").is_file():
        raise RuntimeError("官方启动脚本不存在，检查下载目录")
    print(f"官方代码版本已确认：{result}")


def patch_launcher(source: str, train_subset: int = 0, val_subset: int = 0) -> str:
    """Change orchestration only: serialize two AE runs; optional AE subsets."""
    first = '    ae_pids=()\n'
    last = '    wait_for_jobs "${ae_pids[@]}"\n'
    if source.count(first) != 1 or source.count(last) != 1:
        raise RuntimeError("上游 AE 启动块已变化，请人工检查，拒绝自动修改核心运行流程")
    i = source.index(first)
    j = source.index(last, i) + len(last)
    block = source[i:j]
    runs = re.findall(r'    train_ae [^\n]+ &\n', block)
    if len(runs) != 2:
        raise RuntimeError("未识别到官方两个 AE 训练任务，不执行替换")
    serial = "".join(line[:-3] + "\n" for line in runs)
    result = source[:i] + serial + source[j:]
    for key, value in (("train", train_subset), ("val", val_subset)):
        before = f"data.{key}_subset_size=0"
        if result.count(before) != 1:
            raise RuntimeError(f"找不到 {before}，不能保证数据子集参数准确")
        if value:
            result = result.replace(before, f"data.{key}_subset_size={value}", 1)
    return result


def environment(profile: str, gpu: str, train_dir: Path, val_dir: Path) -> dict:
    if not gpu.isdigit():
        raise ValueError("--gpu 目前接受一个 GPU 编号，例如 0")
    env = os.environ.copy()
    env.update({
        "ROOT_DIR": str(UPSTREAM),
        "TRAIN_DIR": str(train_dir),
        "VAL_DIR": str(val_dir),
        "AE_GPU_VAE": gpu,
        "AE_GPU_GRWM": gpu,
        "PRECOMPUTE_GPUS": gpu,
        "DYNAMICS_GPUS": gpu,
        "PART_GPUS": gpu,
        "WANDB_MODE": "disabled",
    })
    if profile == "debug":
        env.update({
            "AE_EPOCHS": "1",
            "AE_BATCH_SIZE": "4",
            "AE_MICRO_BATCH_SIZE": "4",
            "DYNAMICS_STEPS": "200",
            "DYNAMICS_BATCH_SIZE": "2",
            "NUM_SAMPLES": "16",
            "EVAL_BATCH_SIZE": "2",
            "SAMPLING_TIMESTEPS": "4",
        })
    elif profile != "paper":
        raise ValueError(f"未知配置：{profile}")
    return env


def validate_data(train_dir: Path, val_dir: Path) -> None:
    for name, directory in (("训练", train_dir), ("验证", val_dir)):
        if not (directory / "info.hdf5").is_file() or not (directory / "obs").is_dir():
            raise RuntimeError(
                f"{name}集结构不符合官方格式：{directory}\n"
                "需要 info.hdf5 与 obs/*.npy；请先下载完整官方 Maze9 数据。"
            )


def main() -> int:
    p = argparse.ArgumentParser(description="Maze9 VAE vs GRWM 官方源码复现（单卡串行）")
    p.add_argument("stage", choices=("bootstrap", "check", *STAGES))
    p.add_argument("--train-dir", type=Path, default=os.environ.get("TRAIN_DIR"))
    p.add_argument("--val-dir", type=Path, default=os.environ.get("VAL_DIR"))
    p.add_argument("--profile", choices=("debug", "paper"), default="debug")
    p.add_argument("--gpu", default="0", help="单卡 GPU 编号；默认 0")
    p.add_argument("--dry-run", action="store_true", help="仅显示预期配置，不执行训练")
    args = p.parse_args()

    if args.stage == "bootstrap":
        ensure_source(download=True)
        print("接下来：在 .upstream/official 下安装 requirements.txt，准备 Maze9 数据。")
        return 0
    if os.name == "nt" and not args.dry_run:
        raise RuntimeError("官方 .sh 脚本依赖 Linux；请从 Windows 的 WSL2/Ubuntu 终端运行。")
    train_dir = Path(args.train_dir).expanduser().resolve() if args.train_dir else None
    val_dir = Path(args.val_dir).expanduser().resolve() if args.val_dir else None
    if train_dir is None or val_dir is None:
        p.error("必须指定 --train-dir 和 --val-dir（或设置 TRAIN_DIR / VAL_DIR 环境变量）")
    env = environment(args.profile, args.gpu, train_dir, val_dir)
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "profile": args.profile,
                          "source": str(UPSTREAM), "train": str(train_dir),
                          "val": str(val_dir), "gpu": args.gpu,
                          "overrides": {k: env[k] for k in (
                              "AE_EPOCHS", "DYNAMICS_STEPS", "NUM_SAMPLES") if k in env}},
                          ensure_ascii=False, indent=2))
        return 0
    ensure_source(download=False)
    validate_data(train_dir, val_dir)
    if args.stage == "check":
        print("数据目录与官方代码版本检查通过；依赖和 CUDA 请另用 Python 验证。")
        return 0

    # Official upstream runs both AE trainings concurrently. For a single GPU,
    # change scheduling only (two separate runs in sequence), not the models.
    upstream_script = (UPSTREAM / "run_maze9_vae_grwm.sh").read_text(encoding="utf-8")
    adapted = patch_launcher(upstream_script, 128 if args.profile == "debug" else 0,
                            32 if args.profile == "debug" else 0)
    with tempfile.TemporaryDirectory(prefix="grwm-launch-") as tmp:
        script = Path(tmp) / "run_maze9_single_gpu.sh"
        script.write_text(adapted, encoding="utf-8")
        print(f"运行阶段 {args.stage}，配置 {args.profile}，使用 CUDA GPU {args.gpu}")
        if args.profile == "debug":
            print("注意：debug 只减少 AE 训练子集和训练步数；潜在提取仍处理完整输入目录。")
            print("debug 运行结果不能当作论文报告的真实性能。")
        run_cmd(["bash", str(script), args.stage], cwd=UPSTREAM, env=env)
    print("官方阶段运行完成。使用 python code/export_results.py 导出真实评价结果。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
