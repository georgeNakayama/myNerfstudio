import torch 
from torch import Tensor
import torch.nn.functional as F
from nerfstudio.field_components.activations import trunc_exp
import math


def erf(x, x_squared=None):
    """We use Winitzki's Approximation of the error function """
    if x_squared is None:
        x_squared = x * x 
    x_squared_term = 0.147 * x_squared
    sgn = torch.sign(x)
    frac_term = (4.0 / math.pi + x_squared_term) / (1 + x_squared_term)
    return sgn * torch.sqrt(1 - trunc_exp(-x_squared * frac_term))


def neg_nll_loss(rgb_samples, gt_rgbs, k_samples, eps=1e-5) -> Tensor:
    """
    https://github.com/poetrywanderer/CF-NeRF/blob/main/run_nerf_uncertainty_NF.py#L1031-L1042
    """
    rgb_std = torch.std(rgb_samples, -2, keepdim=True) * k_samples / (k_samples-1) # (N_rays, 3)
    factor = (0.8 / k_samples) ** (-1/7)
    H_sqrt = rgb_std.detach() * factor + eps # (N_rays, 3)
    r_P_C = gaussian_log_prob(gt_rgbs[:, None].expand(-1, k_samples, -1), mean=rgb_samples, variance=H_sqrt * H_sqrt, dim=3).sum(-1)
    loss_nll = - r_P_C.mean()
    return loss_nll
    
    
def gaussian_log_prob(z, mean: float = 0.0, variance: float = 1, dim: int=3) -> Tensor:
    """Return the gaussian probability of z under N(mean, variance)"""
    if not isinstance(mean, Tensor):
        mean = torch.ones_like(z) * mean
    if not isinstance(variance, Tensor):
        variance = torch.ones_like(z) * variance
    assert z.shape[-1] == mean.shape[-1] == variance.shape[-1] == dim 
    main_diff = (z - mean).pow(2) / (2. * variance)
    log_prob =  -main_diff - 0.5 * torch.log(variance) - 0.5 * dim * math.log(2 * math.pi)
    return log_prob
        
    
def gaussian_entropy(logvar, dim = 3) -> Tensor:
    assert logvar.shape[-1] == dim
    logdet = logvar.sum(-1)
    entropy = 0.5 * (dim + dim * math.log(2 * math.pi) + logdet)
    return entropy


def logistic_normal_entropy(samples, mean, logvar, dim = 3) -> Tensor:
    # we upper bound the entropy and minimize the uppder bound 
    # the upper bound is given by logistic normal entropy <= gaussian entropy - int_R |x|P_N(x)dx
    assert samples.shape[-1] == mean.shape[-1] == logvar.shape[-1] == dim
    gauss_entropy = gaussian_entropy(logvar, dim=dim)
    variance = trunc_exp(logvar)
    std = trunc_exp(0.5 * logvar)
    exponent = mean / (math.sqrt(2.) * std)
    second_term = 2. * variance * trunc_exp(-1.0 * exponent * exponent) + math.sqrt(math.pi / 2.) * mean * std * torch.erf(exponent)
    entropy = gauss_entropy - second_term.sum(-1)
    return entropy

def log_normal_entropy(samples, mean, logvar, dim=3) -> Tensor:
    gauss_entropy = gaussian_entropy(logvar, dim=dim)
    return gauss_entropy + mean.sum(-1)