"""Exact invariants refactored from the archived Pendulum implementation."""
import torch


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def freeze_snapshot(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_frozen(model, snapshot):
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError('Frozen model gained gradients or trainable parameters')
    state = model.state_dict()
    if state.keys() != snapshot.keys() or any(not torch.equal(state[name].detach().cpu(), value.detach().cpu()) for name, value in snapshot.items()):
        raise AssertionError('Frozen model weights changed')
