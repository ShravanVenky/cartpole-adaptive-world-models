"""Stable independent stream seeds; no Python process-dependent hash()."""
import hashlib
import random
import numpy as np
import torch


def stable_seed(seed, *namespace):
    message = '/'.join(map(str, (seed, *namespace)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], 'little') % (2**63-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def generator(seed, device='cpu', *namespace):
    return torch.Generator(device=device).manual_seed(stable_seed(seed, *namespace))
