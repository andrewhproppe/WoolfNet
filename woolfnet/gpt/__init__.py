from pathlib import Path

import optax
import torch.optim as optim
from torch import nn

GPT_ARTIFACTS = Path(__file__).parent / "artifacts"
GPT_ARTIFACTS.mkdir(parents=True, exist_ok=True)

GPT_CONFIGS = Path(__file__).parent / "configs"
GPT_CONFIGS.mkdir(parents=True, exist_ok=True)


ACTIVATION_REGISTRY = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    "Identity": nn.Identity,
}

LR_SCHEDULER_REGISTRY = {
    "RLROP": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
}

LR_SCHEDULER_REGISTRY_JAX = {
    "warm_cos": optax.schedules.warmup_cosine_decay_schedule,
    "warm_exp": optax.schedules.warmup_exponential_decay_schedule,
}

LOSS_FN_REGISTRY = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
}
