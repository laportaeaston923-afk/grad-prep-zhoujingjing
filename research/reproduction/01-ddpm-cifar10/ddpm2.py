
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from unet import UNet
from ddpm import DDPM


# =========================
# 1. 加载训练好的模型
# =========================
def load_model(device):

    # 与 train.py 中的 UNet 参数保持一致
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

    model_path = Path(r"E:\all\ddpm - 副本\ddpm.pth")

    # 加载已经训练好的参数
    ddpm.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    ddpm.eval()

    print("模型加载成功！")

    return ddpm


# =========================
# 2. 保存反向去噪中间过程
# =========================
@torch.no_grad()
def sample_trajectory(ddpm, device):

    # 从纯高斯噪声开始
    x = torch.randn(
        (1, 3, 32, 32),
        device=device
    )

    trajectory = []

    # 选择13个显示节点（包含初始噪声）
    save_steps = set(
        torch.linspace(
            0, ddpm.timesteps, 13
        ).round().long().tolist()
    )

    # 保存初始纯噪声
    trajectory.append((
        ddpm.timesteps,
        x[0].cpu().clone()
    ))

    # 执行1000步反向去噪
    for i in reversed(range(ddpm.timesteps)):

        t = torch.full(
            (1,),
            i,
            device=device,
            dtype=torch.long
        )

        # 生成随机噪声
        z = torch.randn_like(x)

        # 调用你原来ddpm.py中的p_sample
        x = ddpm.p_sample(x, t, z)

        # 保存指定时间步
        if i in save_steps:

            trajectory.append((
                i,
                x[0].cpu().clone()
            ))

            print(f"保存时间步: t={i}")

    return trajectory


# =========================
# 3. 绘制去噪轨迹
# =========================
def show_trajectory(trajectory):

    fig, axes = plt.subplots(
        1,
        len(trajectory),
        figsize=(20, 2.5)
    )

    for ax, (t, img) in zip(axes, trajectory):

        # 图像从[-1,1]转换到[0,1]
        img = (img.clamp(-1, 1) + 1) / 2

        # [C,H,W] -> [H,W,C]
        img = img.permute(1, 2, 0).numpy()

        ax.imshow(img)

        ax.set_title(
            f"t={t}",
            fontsize=9
        )

        ax.axis("off")

    plt.suptitle(
        "Reverse Denoising Process: Noise to Image",
        fontsize=13
    )

    plt.tight_layout()

    # 保存到项目目录
    save_path = Path(
        r"E:\all\ddpm - 副本\denoising_trajectory.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("去噪轨迹已保存：", save_path)

    plt.show()
    plt.close(fig)


# =========================
# 4. 主程序
# =========================
if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前设备：", device)

    # 加载模型
    ddpm = load_model(device)

    # 执行反向去噪并保存轨迹
    trajectory = sample_trajectory(
        ddpm,
        device
    )

    # 绘制轨迹图
    show_trajectory(trajectory)
