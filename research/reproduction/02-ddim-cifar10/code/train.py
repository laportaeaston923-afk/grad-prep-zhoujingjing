from pathlib import Path
import torch
from tqdm.auto import tqdm

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from unet import UNet
from ddpm import DDPM

def get_dataset(batch_size=256):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(  (0.5,0.5,0.5),
        (0.5,0.5,0.5))
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
def train(
    epochs: int,
    dataloader: DataLoader,
    ddpm: DDPM,
    device: torch.device
):
    ddpm.train()

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):

        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{epochs}"
        )

        total_loss = 0.0
        total_samples = 0

        for x_0, _ in pbar:

            x_0 = x_0.to(device, non_blocking=True)


            batch_size = x_0.shape[0]

            # ★ 修改1：时间步转成float
            t = torch.randint(
                0,
                ddpm.timesteps,
                (batch_size,),
                device=device
            ).float()

            # ★ 修改2：生成高斯噪声
            eps = torch.randn_like(x_0)

            # 前向扩散
            x_t = ddpm.q_sample(x_0, t.long(), eps)

            # UNet预测噪声
            eps_theta = ddpm.eps_model(x_t, t)

            # MSE损失
            loss = loss_fn(eps_theta, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix(
                avg_loss=f"{total_loss / total_samples:.6f}",
                batch_loss=f"{loss.item():.6f}"
            )

        tqdm.write(
            f"Epoch {epoch + 1} finished, "
            f"avg_loss={total_loss / total_samples:.6f}"
        )


if __name__ == "__main__":

    dataloader = get_dataset()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device =", device)

    if torch.cuda.is_available():
        print("GPU =", torch.cuda.get_device_name(0))
    # ====================

    ddpm_unet = UNet(
        image_channels=3,
        n_channels=64,
        ch_mults=(1, 2, 4),
        is_attn=(False, False, False),
        n_blocks=2
    )
    ddpm = DDPM(
        timesteps=1000,
        eps_model=ddpm_unet
    ).to(device)
    # 检查模型是否在GPU
    print("model device =", next(ddpm.parameters()).device)

    train(
        epochs=100,
        dataloader=dataloader,
        ddpm=ddpm,
        device=device
    )

    output = Path(__file__).resolve().parents[1] / "results" / "ddpm.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ddpm.state_dict(), output)
