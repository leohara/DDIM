import torch
from torchvision import transforms
from tqdm import tqdm


class DDPMDiffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def sample(self, model, x_shape=(20, 1, 28, 28)):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        return x

    def reverse_to_img(self, x):
        x = (x * 0.5 + 0.5) * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        images = [to_pil(x[i]) for i in range(x.size(0))]
        return images


class DDIMDiffuser(DDPMDiffuser):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        super().__init__(num_timesteps, beta_start, beta_end, device)

    def sample(self, model, x_shape=(20, 1, 28, 28), ddim_steps=50):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        ddim_timesteps = torch.linspace(0, self.num_timesteps-1, ddim_steps, dtype=torch.long, device=self.device)

        for i in tqdm(range(ddim_steps-1, -1, -1)):
            t = ddim_timesteps[i]
            t_prev = ddim_timesteps[i-1] if i > 0 else 0
            x = self.denoise_ddim(model, x, t, t_prev)

        return x

    def denoise_ddim(self, model, x, t, t_prev):
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        model.eval()
        with torch.no_grad():
            eps = model(x, t.unsqueeze(0).repeat(x.size(0)).to(x.device))
        model.train()

        x_0_pred = (x - eps * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
        direction = torch.sqrt(1.0 - alpha_bar_prev) * eps
        x_t_prev = torch.sqrt(alpha_bar_prev) * x_0_pred + direction
        return x_t_prev
