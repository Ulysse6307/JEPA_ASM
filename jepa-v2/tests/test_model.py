"""FactoredEncoder forward contract: correct output shapes and robustness to
graphs that are missing one of the three edge relations."""
import torch

from jepa_v2.config import ModelConfig
from jepa_v2.data import Vocab, collate_programs, view_to_data
from jepa_v2.model import FactoredEncoder


def _tiny_cfg(vocab_size):
    return ModelConfig(vocab_size=vocab_size, node_emb_dim=16, hidden_dim=32,
                       num_layers=2, sem_dim=8, speed_dim=4)


def test_forward_returns_factored_shapes():
    v = Vocab.build([["a", "b", "c"]], 32)
    views = [view_to_data(["a", "b", "c"], [(0, 1, 0), (1, 2, 1), (0, 2, 2)], v, l)
             for l in range(4)]
    b = collate_programs([views])            # 1 program, 4 views
    model = FactoredEncoder(_tiny_cfg(v.size)).eval()
    z_sem, z_speed = model(b)
    assert z_sem.shape == (4, 8)
    assert z_speed.shape == (4, 4)
    assert torch.isfinite(z_sem).all() and torch.isfinite(z_speed).all()


def test_forward_handles_missing_relation():
    v = Vocab.build([["a", "b"]], 8)
    # two graphs, only relation 0 present (no data/call edges)
    views = [view_to_data(["a", "b"], [(0, 1, 0)], v, l) for l in range(2)]
    b = collate_programs([views])
    model = FactoredEncoder(_tiny_cfg(v.size)).eval()
    z_sem, z_speed = model(b)
    assert z_sem.shape == (2, 8)
    assert torch.isfinite(z_sem).all()


def test_embed_concatenates_blocks():
    v = Vocab.build([["a", "b", "c"]], 16)
    views = [view_to_data(["a", "b", "c"], [(0, 1, 0), (1, 2, 1)], v, l)
             for l in range(2)]
    b = collate_programs([views])
    cfg = _tiny_cfg(v.size)
    z = FactoredEncoder(cfg).embed(b)
    assert z.shape == (2, cfg.sem_dim + cfg.speed_dim)
