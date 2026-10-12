"""Replot uploaded MATLAB outputs and recompute the official DFS-fit objective.

No CSI preprocessing or BVP optimization is rerun here. This verifies saved
arrays for the bundled position-1/orientation-1 sample (orientation -90 deg).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def rotate_like_author(x: np.ndarray, quarter_turns: int) -> np.ndarray:
    padded = np.pad(x, ((1, 0), (1, 0), (0, 0)))
    return np.rot90(padded, quarter_turns, axes=(0, 1))[1:, 1:, :]


def mapping(o: dict, bins: np.ndarray) -> np.ndarray:
    v = o["velocity_bin"].ravel()
    xx, yy = np.meshgrid(v, v, indexing="ij")
    a = o["A"]
    hz = (a[:, 0, None, None] * xx + a[:, 1, None, None] * yy) / o["wave_length"].item()
    # MATLAB round: half values round away from zero.
    hz = (np.sign(hz) * np.floor(np.abs(hz) + 0.5)).astype(int)
    w = np.zeros((a.shape[0], len(bins), len(v), len(v)))
    for r in range(a.shape[0]):
        for i in range(len(v)):
            for j in range(len(v)):
                f = hz[r, i, j]
                if bins[0] <= f <= bins[-1]:
                    w[r, f - int(bins[0]), i, j] = 1
                else:
                    # Preserve the official implementation's out-of-band penalty.
                    w[r, :, i, j] = 1e10
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    result = base.parent / "results" / "userA-1-1-1-1"
    ap.add_argument("--dfs", type=Path, default=result / "DFS.mat")
    ap.add_argument("--bvp", type=Path, default=result / "BVP.mat")
    ap.add_argument("--reference", type=Path, default=base / "BVP" /
                    "user-user-1-1-1-1-1-1e-07-100-20-100000-L0.mat")
    ap.add_argument("--out", type=Path, default=result)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s, o, r = loadmat(args.dfs), loadmat(args.bvp), loadmat(args.reference)
    dfs = s["doppler_spectrum"]
    ours, ref = o["velocity_spectrum_ro"], r["velocity_spectrum_ro"]
    bins, v = s["freq_bin"].ravel(), o["velocity_bin"].ravel()
    if dfs.shape[:2] != (6, 121) or not np.array_equal(bins, np.arange(-60, 61)):
        raise ValueError("Expected six links and sorted -60:60 Hz bins.")
    if ours.shape != ref.shape or ours.shape[:2] != (20, 20):
        raise ValueError("BVP dimensions must agree; no temporal cropping is performed.")
    for x in (dfs, ours, ref):
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("Invalid or negative power values.")
    if o["torso_orientation_deg"].item() != -90:
        raise ValueError("Reference inverse-rotation check supports this -90 deg sample only.")
    if not np.array_equal(rotate_like_author(o["velocity_spectrum"], -1), ours):
        raise ValueError("Saved raw and rotated BVP do not agree.")
    ref_raw = rotate_like_author(ref, 1)
    if not np.array_equal(rotate_like_author(ref_raw, -1), ref):
        raise ValueError("Reference rotation is not losslessly invertible for this array.")

    fs, segment_len = s["sample_rate_hz"].item(), int(o["seg_length"].item())
    time = s["time_s"].ravel()
    segments = ours.shape[2]
    target = dfs[:, :, :segments * segment_len].reshape(6, 121, segments, segment_len).mean(3)
    mass = target.sum(1)
    if (mass <= 0).any():
        raise ValueError("Missing or zero-mass DFS link in a segment.")
    target *= mass[0][None, None, :] / mass[:, None, :]
    w = mapping(o, bins)
    proj_ours = np.einsum("rfij,ijt->rft", w, o["velocity_spectrum"])
    proj_ref = np.einsum("rfij,ijt->rft", w, ref_raw)
    emd_ours = np.abs(np.cumsum(proj_ours - target, axis=1)).sum((0, 1))
    emd_ref = np.abs(np.cumsum(proj_ref - target, axis=1)).sum((0, 1))
    recomputed = emd_ours + o["lambda"].item() * np.count_nonzero(o["velocity_spectrum"], axis=(0, 1))
    obj_error = np.max(np.abs(recomputed - o["objectives"].ravel()))
    if obj_error > 1e-8:
        raise ValueError(f"Objective recomputation does not match the saved solver output: {obj_error}")

    flat_o, flat_r = ours.reshape(400, segments), ref.reshape(400, segments)
    rel = np.linalg.norm(flat_o - flat_r, axis=0) / np.linalg.norm(flat_r, axis=0)
    cosine = np.sum(flat_o * flat_r, axis=0) / (np.linalg.norm(flat_o, axis=0) * np.linalg.norm(flat_r, axis=0))
    overlap = np.minimum(flat_o / flat_o.sum(0), flat_r / flat_r.sum(0)).sum(0)
    diagnostics = {
        "sample": str(s["sample"][0]), "dfs_shape": list(dfs.shape),
        "bvp_shape": list(ours.shape), "sample_rate_assumed_hz": fs,
        "segment_length_snapshots": segment_len, "discarded_DFS_snapshots": dfs.shape[2] - segments * segment_len,
        "solver_seconds": o["bvp_seconds"].item(), "all_exit_flags_one": bool((o["exit_flags"] == 1).all()),
        "mean_relative_L2_BVP": float(rel.mean()), "mean_cosine_similarity_BVP": float(cosine.mean()),
        "mean_normalized_overlap_BVP": float(overlap.mean()),
        "mean_DFS_EMD_ours": float(emd_ours.mean()), "mean_DFS_EMD_candidate_reference": float(emd_ref.mean()),
        "segments_ours_lower_EMD": int((emd_ours < emd_ref - 1e-7).sum()),
        "segments_reference_lower_EMD": int((emd_ref < emd_ours - 1e-7).sum()),
        "segments_EMD_tied_within_1e-7": int((np.abs(emd_ours - emd_ref) <= 1e-7).sum()),
        "max_saved_objective_recomputation_error": float(obj_error),
        "segment13_max_abs_BVP_difference": float(np.max(np.abs(ours[:, :, 12] - ref[:, :, 12]))),
        "candidate_reference_sample_identity_confirmed": False,
        "plot_provenance": "Replotted in Python from user-uploaded MATLAB arrays; no MATLAB optimization rerun.",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.dfs, args.bvp, args.reference)},
    }
    (args.out / "verification_summary.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "DFS_fit_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "ours_EMD", "candidate_reference_EMD", "ours_minus_reference"])
        writer.writerows((i + 1, emd_ours[i], emd_ref[i], emd_ours[i] - emd_ref[i]) for i in range(segments))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), constrained_layout=True)
    for rx, ax in enumerate(axes.flat):
        im = ax.imshow(dfs[rx], origin="lower", aspect="auto", cmap="viridis",
                       extent=[time[0], time[-1], -60, 60], vmin=0, vmax=dfs.max())
        ax.set(title=f"Receiver {rx + 1}", xlabel="Nominal time (s)", ylabel="Doppler frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Normalized power")
    fig.savefig(args.out / "DFS_6_receivers.png", dpi=140)
    plt.close(fig)

    selected = np.unique(np.round(np.linspace(0, segments - 1, 6)).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.3), constrained_layout=True)
    for idx, ax in zip(selected, axes.flat):
        im = ax.imshow(ours[:, :, idx].T, origin="lower", cmap="viridis",
                       extent=[v[0] - .1, v[-1] + .1, v[0] - .1, v[-1] + .1], vmin=0, vmax=ours.max())
        ax.set(title=f"Segment {idx + 1}, t = {o['segment_time_s'].ravel()[idx]:.2f} s",
               xlabel="Body vx (m/s)", ylabel="Body vy (m/s)")
        fig.colorbar(im, ax=ax, label="Power")
    fig.savefig(args.out / "BVP_snapshots.png", dpi=140)
    plt.close(fig)

    selected = np.unique(np.round(np.linspace(0, segments - 1, 4)).astype(int))
    diff = np.abs(ours - ref)
    fig, axes = plt.subplots(len(selected), 3, figsize=(9.5, 11), constrained_layout=True)
    vmax = max(ours.max(), ref.max())
    for row, idx in enumerate(selected):
        for col, (x, title) in enumerate(zip((ref, ours, diff), ("Candidate reference", "Your reproduction", "Absolute difference"))):
            ax = axes[row, col]
            im = ax.imshow(x[:, :, idx].T, origin="lower", cmap="viridis",
                           extent=[v[0] - .1, v[-1] + .1, v[0] - .1, v[-1] + .1],
                           vmin=0, vmax=vmax if col < 2 else max(diff.max(), 1e-12))
            ax.set(title=f"{title} | segment {idx + 1}", xlabel="Body vx (m/s)", ylabel="Body vy (m/s)")
            fig.colorbar(im, ax=ax)
    fig.savefig(args.out / "BVP_reference_comparison.png", dpi=110)
    plt.close(fig)

    x = np.arange(1, segments + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    axes[0].plot(x, rel, "o-", label="Relative L2 error")
    axes[0].set(xlabel="Segment", ylabel="Relative L2 error", title="BVP values: lower error means closer values")
    axes[1].plot(x, cosine, "o-", label="Cosine similarity")
    axes[1].plot(x, overlap, "s-", label="Normalized overlap")
    axes[1].set(xlabel="Segment", ylabel="Similarity", ylim=(0, 1.05), title="BVP distribution comparison")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(args.out / "BVP_comparison_metrics.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    ax.plot(x, emd_ours, "o-", label="Your reproduction")
    ax.plot(x, emd_ref, "s-", label="Bundled candidate reference")
    ax.set(xlabel="Segment", ylabel="Official EMD term (lower is better)", title="Fit to the same uploaded DFS")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(args.out / "DFS_fit_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
