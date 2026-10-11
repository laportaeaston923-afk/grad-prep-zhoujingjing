import torch
from torch import nn

class DDPM(nn.Module):
    def __init__(self,timesteps:int,eps_model:nn.Module):
        super().__init__()
        self.timesteps = timesteps
        self.eps_model = eps_model

        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
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

    def p_sample(self, x_t, t, z):
        eps_theta = self.eps_model(x_t, t.float())

        alpha = self.alpha[t][:, None, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        mean = (x_t- (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta) / torch.sqrt(alpha)

        # t=0时不再加噪声
        noise = (t > 0).float()[:, None, None, None]

        return mean + noise * torch.sqrt(beta) * z
    @torch.no_grad()
    def sample(self, shape, device):
        x = torch.randn(shape, device=device)
        samples = []

        # T -> 0
        for i in reversed(range(self.timesteps)):

            t = torch.full(
                (shape[0],),
                i,
                device=device,
                dtype=torch.long
            )

            z = torch.randn_like(x)

            x = self.p_sample(x, t, z)

        return x
    @torch.no_grad()
    #从1000个里面挑50个
    def ddim_sample(self, shape, device, steps=50, eta=0.0):

        x = torch.randn(shape, device=device)

        times = torch.linspace(
            self.timesteps - 1,
            0,
            steps,
            device=device
        ).long()


        for i in range(len(times)-1):

            t = times[i]
            t_next = times[i+1]

            t_batch = torch.full(
                (shape[0],),
                t,
                device=device,
                dtype=torch.long
            )

            #预测噪声
            eps_theta = self.eps_model(
                x,
                t_batch.float()
            )


            alpha_bar_t = self.alpha_bar[t]

            alpha_bar_next = self.alpha_bar[t_next]


            x0 = (
                x -
                torch.sqrt(1-alpha_bar_t) * eps_theta
            ) / torch.sqrt(alpha_bar_t)


            sigma = eta * torch.sqrt(
                (1-alpha_bar_next)
                /
                (1-alpha_bar_t)
                *
                (1-alpha_bar_t/alpha_bar_next)
            )


            noise = torch.randn_like(x)


            direction = torch.sqrt(
                1-alpha_bar_next-sigma**2
            ) * eps_theta

            #ddim反向公式
            x = (
                torch.sqrt(alpha_bar_next)*x0
                +
                direction
                +
                sigma*noise
            )


        return x