"""Sequential support, held-out trajectories and scattered reference data."""
from .collection import collect_iid, collect_sequential
from .transitions import TransitionBatch, assert_disjoint

__all__ = ["TransitionBatch", "assert_disjoint", "collect_sequential", "collect_iid"]
