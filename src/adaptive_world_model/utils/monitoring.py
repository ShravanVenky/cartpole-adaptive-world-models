"""Reusable error monitor from V0, deliberately disconnected from CartPole V1."""
from dataclasses import dataclass
import math


@dataclass
class PersistentErrorMonitor:
    threshold: float
    consecutive: int = 3
    decay: float = 0.8
    average: float | None = None
    count: int = 0

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold <= 0 or type(self.consecutive) is not int or self.consecutive < 1 or not 0 <= self.decay < 1:
            raise ValueError('Invalid persistent error monitor settings')

    def reset(self):
        self.average = None; self.count = 0

    def observe(self, error):
        if not math.isfinite(error) or error < 0: raise ValueError('Error must be finite and nonnegative')
        self.average = error if self.average is None else self.decay*self.average+(1-self.decay)*error
        self.count = self.count+1 if self.average > self.threshold else 0
        return self.count >= self.consecutive
