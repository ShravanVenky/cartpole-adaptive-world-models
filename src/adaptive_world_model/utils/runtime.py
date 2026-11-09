"""Actual execution device, versions and future run deadlines."""
import importlib.metadata
import platform
import sys
import time
import torch


class Deadline:
    def __init__(self, minutes):
        self.start = time.perf_counter(); self.end = self.start+minutes*60

    def check(self, reserve_seconds=30):
        if time.perf_counter() >= self.end-reserve_seconds:
            raise TimeoutError('Configured stage deadline reached; saved outputs remain incomplete')


def select_device(requested='cuda'):
    if requested not in ['cpu', 'cuda']: raise ValueError('Device must be explicitly cpu or cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; select CPU explicitly for structural tests')
    return torch.device(requested)


def hardware_info(device):
    result = {'device':str(device), 'torch':torch.__version__, 'cuda_runtime':torch.version.cuda,
              'python':sys.version, 'executable':sys.executable, 'platform':platform.platform(),
              'versions':{name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','matplotlib']}}
    if str(device).startswith('cuda'):
        result['gpu'] = torch.cuda.get_device_name(device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(vram_free_bytes=free, vram_total_bytes=total)
    return result
