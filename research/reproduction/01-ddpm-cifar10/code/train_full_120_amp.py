"""Resume CIFAR-10 DDPM training with CUDA AMP, keeping originals untouched.

Source checkpoint: ../results/train_results_120/checkpoint.pth
AMP outputs:       ../results/train_results_120_amp/

Run from code/ containing ddpm.py and unet.py:
    python train_full_120_amp.py

Every epoch the AMP-specific checkpoint saves model + Adam + GradScaler +
the combined (pre-AMP and AMP) REAL epoch-average training loss history.
"""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from ddpm import DDPM
from unet import UNet


# Keep the same model, data preprocessing and optimizer hyperparameters
# as train_full_120.py so that the existing checkpoint can be resumed.
END_EPOCH = 120
BATCH_SIZE = 256
LEARNING_RATE = 2e-4

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
SOURCE_CHECKPOINT = RESULTS_DIR / "train_results_120" / "checkpoint.pth"
OUTPUT_DIR = RESULTS_DIR / "train_results_120_amp"
AMP_CHECKPOINT = OUTPUT_DIR / "checkpoint_amp.pth"
LOSS_CSV = OUTPUT_DIR / "training_loss.csv"
LOSS_FIG = OUTPUT_DIR / "training_loss.png"
FINAL_WEIGHTS = OUTPUT_DIR / "ddpm_120_epochs_amp.pth"


def create_dataloader(device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(
        root=str(CODE_DIR / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Same as original; change later only after benchmarking.
        pin_memory=(device.type == "cuda"),
    )


def create_model(device):
    unet = UNet(
        image_channels=3,
        n_channels=64,
        ch_mults=(1, 2, 4),
        is_attn=(False, False, False),
        n_blocks=2,
    )
    return DDPM(timesteps=1000, eps_model=unet).to(device)


def load_for_resume(model, optimizer, scaler):
    # Once the AMP run has begun, resume its OWN newer checkpoint.
    # Otherwise import the original FP32 checkpoint, read-only.
    if AMP_CHECKPOINT.is_file():
        checkpoint_path = AMP_CHECKPOINT
        from_amp = True
    elif SOURCE_CHECKPOINT.is_file():
        checkpoint_path = SOURCE_CHECKPOINT
        from_amp = False
    else:
        raise FileNotFoundError(
            "No checkpoint found. Expected:\n"
            f"  {SOURCE_CHECKPOINT}\n"
            f"or\n  {AMP_CHECKPOINT}"
        )

    # map_location='cpu' avoids an extra peak in GPU memory when loading.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    required_keys = ("epoch", "model", "optimizer", "history")
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint missing keys: {missing}")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if from_amp and scaler.is_enabled() and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    completed_epoch = int(checkpoint["epoch"])
    history = [(int(ep), float(loss)) for ep, loss in checkpoint["history"]]
    # An unbroken loss history is crucial for the README curve.
    actual_epochs = [ep for ep, _ in history]
    expected_epochs = list(range(1, completed_epoch + 1))
    if actual_epochs != expected_epochs:
        raise ValueError(
            "Checkpoint history is not complete from Epoch 1 through "
            f"Epoch {completed_epoch}; refusing to mislabel an incomplete curve."
        )

    print(f"Loaded checkpoint (read only): {checkpoint_path}")
    print(f"Completed epochs: {completed_epoch}")
    print(f"Preserved REAL loss records: {len(history)}")
    print(f"Next epoch: {completed_epoch + 1}/{END_EPOCH}")
    del checkpoint
    return completed_epoch + 1, history


def save_loss_results(history):
    # This ONLY writes to OUTPUT_DIR, never to train_results_120/.
    with LOSS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_mse_loss"])
        writer.writerows(history)

    epochs = [epoch for epoch, _ in history]
    losses = [loss for _, loss in history]
    smooth_losses = [
        sum(losses[max(0, i - 4):i + 1]) /
        len(losses[max(0, i - 4):i + 1])
        for i in range(len(losses))
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs, losses, color="#93b4df", linewidth=1.0,
        alpha=0.65, label="Actual epoch-average training loss",
    )
    plt.plot(
        epochs, smooth_losses, color="#2563eb", linewidth=2.4,
        label="Moving average (5 epochs)",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE noise-prediction loss")
    plt.title("DDPM on CIFAR-10: Training Loss (FP32 + AMP)")
    plt.xlim(1, END_EPOCH)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_FIG, dpi=300, bbox_inches="tight")
    plt.close()


def save_amp_checkpoint(epoch, model, optimizer, scaler, history):
    # Temp file + atomic replace protects the previous AMP checkpoint
    # if saving is interrupted. Does NOT touch the original FP32 file.
    temp_path = OUTPUT_DIR / "checkpoint_amp.tmp"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
        },
        temp_path,
    )
    temp_path.replace(AMP_CHECKPOINT)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda")

    print("Device:", device)
    if amp_enabled:
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    print("AMP FP16 enabled:", amp_enabled)
    print("Original checkpoint will NOT be overwritten.")
    print("AMP output directory:", OUTPUT_DIR)

    loader = create_dataloader(device)
    model = create_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch, history = load_for_resume(model, optimizer, scaler)
    # Copy the pre-AMP loss history into the new output folder too.
    if history:
        save_loss_results(history)

    for epoch in range(start_epoch, END_EPOCH + 1):
        model.train()
        # Accumulate on GPU; avoid loss.item() synchronizing CUDA each batch.
        total_loss_gpu = torch.zeros((), device=device)
        total_samples = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{END_EPOCH}", total=len(loader))

        for batch_index, (x_0, _) in enumerate(pbar, start=1):
            x_0 = x_0.to(device, non_blocking=True)
            batch_size = x_0.size(0)
            t = torch.randint(0, model.timesteps, (batch_size,), device=device)
            eps = torch.randn_like(x_0)

            # Keep forward-diffusion arithmetic in FP32 as before.
            x_t = model.q_sample(x_0, t, eps)
            optimizer.zero_grad(set_to_none=True)

            # Only run the U-Net forward pass under autocast.
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                eps_theta = model.eps_model(x_t, t.float())

            # FP32 loss accumulation for numerical stability and meaningful MSE.
            loss = F.mse_loss(eps_theta.float(), eps)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_gpu += loss.detach() * batch_size
            total_samples += batch_size
            if batch_index % 20 == 0 or batch_index == len(loader):
                running_loss = (total_loss_gpu / total_samples).item()
                if not math.isfinite(running_loss):
                    raise RuntimeError(
                        f"Non-finite training loss in Epoch {epoch}; "
                        "the original checkpoint remains untouched."
                    )
                pbar.set_postfix(avg_loss=f"{running_loss:.6f}")

        avg_loss = (total_loss_gpu / total_samples).item()
        history.append((epoch, avg_loss))
        tqdm.write(f"Epoch {epoch}/{END_EPOCH} | Average Loss: {avg_loss:.6f}")

        # Save the recoverable model state FIRST, then CSV and figure.
        save_amp_checkpoint(epoch, model, optimizer, scaler, history)
        save_loss_results(history)
        tqdm.write(f"Saved AMP checkpoint and real loss curve at Epoch {epoch}")

    if start_epoch <= END_EPOCH:
        torch.save(model.state_dict(), FINAL_WEIGHTS)
        print("Training completed! Final AMP model:", FINAL_WEIGHTS)
    else:
        print("Checkpoint already covers all 120 epochs; no training needed.")
    print("Loss CSV:", LOSS_CSV)
    print("Loss figure:", LOSS_FIG)
    print("AMP checkpoint:", AMP_CHECKPOINT)


if __name__ == "__main__":
    main()
