import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class GMM():
    def __init__(
        self,
        num_components: int,
        dim: int,
        mean_scale: float = 4.0,
        cov_scale: float = 0.5,
        min_separation: float = None,  # None => auto from cov_scale
        max_tries: int = 10_000,
        device: str = "cpu",
    ):
        assert dim in (2, 3), "dim must be 2 or 3"
        self.name = "gmm"
        self.num_components = num_components
        self.dim = dim
        self.device = device

        # Component radius is roughly sqrt(cov_scale + 0.5); require a few of those
        # between means so the blobs are visually distinct.
        if min_separation is None:
            min_separation = 3.0 * (cov_scale + 0.5) ** 0.5

        means = torch.empty(num_components, dim, device=device)
        placed, tries = 0, 0
        while placed < num_components:
            if tries > max_tries:
                raise RuntimeError(
                    f"Couldn't place {num_components} means with "
                    f"min_separation={min_separation:.2f}. "
                    f"Lower min_separation or raise mean_scale."
                )
            cand = mean_scale * torch.randn(dim, device=device)
            if placed == 0 or torch.linalg.norm(means[:placed] - cand, dim=-1).min() >= min_separation:
                means[placed] = cand
                placed += 1
            tries += 1

        means -= means.mean(0, keepdim=True)

        covs = []
        for _ in range(num_components):
            A = torch.randn(dim, dim, device=device)
            covs.append(cov_scale * (A @ A.T) + 0.5 * torch.eye(dim, device=device))
        covs = torch.stack(covs)

        self.dist = MultivariateNormal(loc=means, covariance_matrix=covs)

    def sample(self, batch_size, return_labels=False):
        samples_per_component = batch_size // self.num_components
        x = self.dist.sample((samples_per_component,))
        x = x.reshape(-1, self.dim)
        labels = torch.arange(
            self.num_components,
            device=self.device
        ).repeat(samples_per_component)
        if return_labels:
            return x, labels
        else:
            return x
        

class Circle:
    def __init__(self, radius: float = 1.0, device: str = "cpu"):
        self.name = "circle"
        self.radius = radius
        self.dim = 2
        self.device = device

    def sample(self, batch_size, return_labels=False):
        theta = 2 * torch.pi * torch.rand(batch_size, device=self.device)
        x = self.radius * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        if return_labels:
            return x, None
        else:
            return x
        

class Plane:
    def __init__(self, low: float = -1.0, high: float = 1.0, device: str = "cpu"):
        self.name = "plane"
        self.low = low
        self.high = high
        self.dim = 2
        self.device = device

    def sample(self, batch_size, return_labels=False):
        x = self.low + (self.high - self.low) * torch.rand(batch_size, self.dim, device=self.device)
        if return_labels:
            return x, None
        return x



class Disk:
    def __init__(self, radius: float = 1.0, device: str = "cpu"):
        self.name = "disk"
        self.radius = radius
        self.dim = 2
        self.device = device

    def sample(self, batch_size, return_labels=False):
        # r = R * sqrt(u) gives uniform area density (naive uniform r concentrates at center)
        u = torch.rand(batch_size, device=self.device)
        r = self.radius * torch.sqrt(u)
        theta = 2 * torch.pi * torch.rand(batch_size, device=self.device)
        x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
        if return_labels:
            return x, None
        return x


class Annulus:
    def __init__(self, inner_radius: float = 0.5, outer_radius: float = 1.0, device: str = "cpu"):
        self.name = "annulus"
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.dim = 2
        self.device = device

    def sample(self, batch_size, return_labels=False):
        # Inverse-CDF for area-uniform radial sampling on [r_in, r_out]
        u = torch.rand(batch_size, device=self.device)
        r = torch.sqrt(u * (self.outer_radius**2 - self.inner_radius**2) + self.inner_radius**2)
        theta = 2 * torch.pi * torch.rand(batch_size, device=self.device)
        x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
        if return_labels:
            return x, None
        return x


class TwoMoons:
    def __init__(self, radius: float = 1.0, noise: float = 0.0, device: str = "cpu"):
        self.name = "two_moons"
        self.radius = radius
        self.noise = noise
        self.dim = 2
        self.device = device

    def sample(self, batch_size: int, return_labels: bool=False):
        # Random moon assignment (label 0 = upper, 1 = lower)
        labels = (torch.rand(batch_size, device=self.device) < 0.5).long()
        # Uniform along the arc (theta in [0, pi] gives uniform arc-length on a unit circle)
        theta = torch.pi * torch.rand(batch_size, device=self.device)

        # Upper moon: upper half of unit circle centered at (-0.5, -0.25)
        # Lower moon: lower half of unit circle centered at ( 0.5,  0.25)
        # Shift chosen so the two-moons figure is centered at the origin.
        upper_x = torch.cos(theta) - 0.5
        upper_y = torch.sin(theta) - 0.25
        lower_x = 0.5 - torch.cos(theta)
        lower_y = 0.25 - torch.sin(theta)

        x_coord = torch.where(labels == 0, upper_x, lower_x)
        y_coord = torch.where(labels == 0, upper_y, lower_y)
        x = self.radius * torch.stack([x_coord, y_coord], dim=-1)

        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)

        if return_labels:
            return x, labels
        return x