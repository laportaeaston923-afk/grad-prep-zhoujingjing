import torch
#显示图片
import matplotlib.pyplot as plt

from ddpm import DDPM
from unet import UNet


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 创建UNet（参数必须和训练时一致）
    unet = UNet(
        image_channels=3,
        n_channels=64,
        ch_mults=(1, 2, 4),
        is_attn=(False, False, False),
        n_blocks=2
    )

    ddpm = DDPM(
        timesteps=1000,
        eps_model=unet
    ).to(device)

    # 加载训练好的模型
    ddpm.load_state_dict(
        torch.load(
            r"E:\all\ddpm - 副本\ddpm.pth",
            map_location=device
        )
    )

    ddpm.eval()

    # 采样生成16张图片
    imgs = ddpm.sample(
        shape=(16, 3, 32, 32),
        device=device
    )

    # 恢复到[0,1]
    imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)

    # 显示结果
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))

    for i in range(16):
        #每个图片具体的位置
        ax = axes[i // 4][i % 4]

        img = imgs[i].permute(1, 2, 0)  # (3,32,32) -> (32,32,3)
        #重新排列三个维度

        ax.imshow(
            img.cpu()
        )

        ax.axis("off")
        #关闭当前子图的坐标轴显示
    #自动调整子图间的间距
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()