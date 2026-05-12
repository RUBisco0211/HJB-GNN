from .base import MultiAgentController
from .hjb_gnn import HJBGNN


def make_algo(algo: str, **kwargs) -> MultiAgentController:
    if algo == 'hjb_gnn':
        return HJBGNN(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
