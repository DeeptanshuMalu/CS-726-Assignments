import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt


class NoiseScheduler:
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model

    """

    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(
                f"{type} scheduler is not implemented"
            )  # change this if you implement additional schedulers

    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(
            beta_start, beta_end, self.num_timesteps, dtype=torch.float32
        )

        self.alphas = torch.cumprod(1 - self.betas, dim=0)

    def __len__(self):
        return self.num_timesteps


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim

    def forward(self, time):
        device = time.device
        half_dim = self.n_dim // 2
        embeddings = torch.exp(
            -torch.linspace(0, torch.log(torch.tensor(10000)), half_dim, device=device)
            / half_dim
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# Reference: https://github.com/albarji/toy-diffusion/blob/master/swissRoll.ipynb


class DiffusionBlock(nn.Module):
    def __init__(self, nunits):
        super(DiffusionBlock, self).__init__()
        self.linear = nn.Linear(nunits, nunits)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = nn.functional.relu(x)
        return x


class DDPM(nn.Module):
    def __init__(self, n_dim=2, n_steps=200):
        super(DDPM, self).__init__()

        nunits = 128
        nblocks = 5
        time_embed_dim = 16

        self.n_dim = n_dim
        self.n_steps = n_steps
        self.inblock = nn.Linear(n_dim + time_embed_dim, nunits)
        self.midblocks = nn.ModuleList([DiffusionBlock(nunits) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, n_dim)

        self.time_embed = SinusoidalPositionEmbeddings(time_embed_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embed(t).view(-1, self.time_embed.n_dim)
        val = torch.hstack([x, time_embed])  # Add t to inputs
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.outblock(val)
        return val


class ConditionalDDPM(nn.Module):
    def __init__(self, n_dim=2, n_steps=200):
        super(ConditionalDDPM, self).__init__()

        nunits = 128
        nblocks = 5
        time_embed_dim = 16
        class_embed_dim = 16

        self.n_dim = n_dim
        self.n_steps = n_steps
        self.inblock = nn.Linear(n_dim + time_embed_dim + class_embed_dim, nunits)
        self.midblocks = nn.ModuleList([DiffusionBlock(nunits) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, n_dim)

        self.time_embed = SinusoidalPositionEmbeddings(time_embed_dim)
        self.class_embed = SinusoidalPositionEmbeddings(class_embed_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, class_label: torch.Tensor
    ) -> torch.Tensor:
        class_embed = self.class_embed(class_label).view(-1, self.class_embed.n_dim)
        time_embed = self.time_embed(t).view(-1, self.time_embed.n_dim)
        val = torch.hstack([x, time_embed, class_embed])
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.outblock(val)
        return val


class ClassifierDDPM:
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """

    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        pass

    def __call__(self, x):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass


def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """

    print(model)
    print(run_name)

    num_timesteps = len(noise_scheduler)
    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, num_timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            noise.requires_grad = True
            alpha_bar_t = noise_scheduler.alphas.to(device)[t].view(-1, 1)
            optimizer.zero_grad()
            xt = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
            t = t.float().reshape(-1, 1)
            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_count += 1
        print(f"Epoch {epoch}: Loss {total_loss/batch_count}")
        losses.append(total_loss / batch_count)
    plt.plot(losses)
    plt.savefig(f"{run_name}/train_loss.png")
    torch.save(model.state_dict(), f"{run_name}/model.pth")


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False):
    """
    Sample from the model

    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """

    n_dim = model.n_dim
    init_sample = torch.randn((n_samples, n_dim)).to(device)
    T = len(noise_scheduler)
    all_samples = []
    for t in range(T - 1, -1, -1):
        t_batch = torch.ones(n_samples).to(device) * t
        alpha_t = 1 - noise_scheduler.betas.to(device)[t].view(-1, 1)
        alpha_bar_t = noise_scheduler.alphas.to(device)[t].view(-1, 1)
        z = torch.randn_like(init_sample, device=device)
        t_batch = t_batch.float().reshape(-1, 1)
        init_sample = (
            1
            / torch.sqrt(alpha_t)
            * (
                init_sample
                - (1 - alpha_t)
                / torch.sqrt(1 - alpha_bar_t)
                * model(init_sample, t_batch)
            )
        )
        if t > 0:
            sigma_t = torch.sqrt(1 - alpha_t)
            init_sample = init_sample + z * sigma_t
        all_samples.append(init_sample.clone())

    if init_sample.shape[1] == 2:
        x1 = init_sample[:, 0].cpu().numpy()
        x2 = init_sample[:, 1].cpu().numpy()
        plt.scatter(x1, x2, s=1)
        plt.title(f"Samples at t={T}")
        plt.savefig(f"{run_name}/samples_{T}.png")
        plt.close()
    elif init_sample.shape[1] == 3:
        x1 = init_sample[:, 0].cpu().numpy()
        x2 = init_sample[:, 1].cpu().numpy()
        x3 = init_sample[:, 2].cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x1, x2, x3, s=1)
        plt.title(f"Samples at t={T}")
        plt.savefig(f"{run_name}/samples_{T}.png")
        plt.close()

    subsample_size = 250
    emd_list = []
    for i in range(5):
        subsample_data_X = utils.sample(data_X.to("cpu").numpy(), size=subsample_size)
        subsample_samples = utils.sample(
            init_sample.to("cpu").numpy(), size=subsample_size
        )
        emd = utils.get_emd(subsample_data_X, subsample_samples)
        print(f"{i} EMD {emd}")
        emd_list.append(emd)
    emd_avg = sum(emd_list) / len(emd_list)
    print(f"EMD: {emd_avg}")

    nll = utils.get_nll(data_X, init_sample)
    print(f"NLL: {nll}")

    with open(f"{run_name}/metrics.txt", "w") as f:
        f.write(f"EMD: {emd_avg}\n")
        f.write(f"NLL: {nll}\n")

    if return_intermediate:
        return all_samples
    else:
        return init_sample


def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass


def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "sample"], default="sample")
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}"  # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(
        num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta
    )
    model = model.to(device)

    if args.mode == "train":
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        if data_y == None:
            data_y = torch.zeros(data_X.shape[0])
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_X, data_y),
            batch_size=args.batch_size,
            shuffle=True,
        )
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == "sample":
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        if data_y == None:
            data_y = torch.zeros(data_X.shape[0])
        model.load_state_dict(torch.load(f"{run_name}/model.pth"))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f"{run_name}/samples_{args.seed}_{args.n_samples}.pth")
    else:
        raise ValueError(f"Invalid mode {args.mode}")
