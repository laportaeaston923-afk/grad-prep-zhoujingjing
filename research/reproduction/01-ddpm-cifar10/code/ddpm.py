import torch
from torch import nn

class DDPM(nn.Module):
    def __init__(self,timesteps:int,eps_model:nn.Module):
        super().__init__()
        self.timesteps = timesteps
        self.eps_model = eps_model

        beta = torch.linspace(1e-4, 0.02, timesteps)#1×10-4到 0.02 之间，均匀生成 timesteps 个数
        alpha = 1 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)#累计相乘
        #注册参数
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
    #前向过程
    def q_sample(self,x_0:torch.Tensor,t:torch.Tensor,eps:torch.Tensor)->torch.Tensor:
        mean = torch.sqrt(self.alpha_bar[t])[:,None,None,None]*x_0#扩展通道
        var = (1-self.alpha_bar[t])[:,None,None,None]
        x_t = mean + torch.sqrt(var)*eps
        return x_t
    #反向过程
    def p_sample(self, x_t, t, z):
        eps_theta = self.eps_model(x_t, t.float())

        alpha = self.alpha[t][:, None, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        mean = (
                       x_t
                       - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
               ) / torch.sqrt(alpha)

        # t=0时不再加噪声
        noise = (t > 0).float()[:, None, None, None]

        return mean + noise * torch.sqrt(beta) * z#方差为简化版的公式
    @torch.no_grad()
    def sample(self, shape, device):
        x = torch.randn(shape, device=device)#生成高斯噪声，xT初始化状态
        samples = []

        # 反向循环T -> 0
        for i in reversed(range(self.timesteps)):

            t = torch.full(
                (shape[0],),
                i,
                device=device,
		#长整型
                dtype=torch.long
            )

            z = torch.randn_like(x)#生成随机噪声

            x = self.p_sample(x, t, z)

        return x
