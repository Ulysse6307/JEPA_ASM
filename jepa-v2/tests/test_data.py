"""Data pipeline: vocab, nx->PyG view building, and the multi-relation batch
offsetting that ProgramData.__inc__ must get right (the encoder depends on it)."""
import torch

from jepa_v2.data import (
    UNK, Vocab, collate_programs, view_to_data,
)


def test_vocab_unk_and_topk():
    v = Vocab.build([["a", "b", "a"], ["b", "c"]], max_size=10)
    assert v.stoi["<unk>"] == UNK == 0
    ids = v.encode(["a", "unseen"])
    assert ids[1].item() == UNK          # OOV -> unk
    assert ids[0].item() != UNK
    assert v.size >= 4


def test_vocab_respects_max_size():
    toks = [[f"t{i}"] for i in range(100)]
    v = Vocab.build(toks, max_size=8)
    assert v.size <= 8


def test_view_to_data_shapes_and_relations():
    texts = ["alloca", "add", "ret"]
    edges = [(0, 1, 0), (1, 2, 1)]           # one control, one data, no call
    v = Vocab.build([texts], 32)
    d = view_to_data(texts, edges, v, lvl=2)
    assert d.x.shape[0] == 3
    assert d.edge_index_0.shape == (2, 1)
    assert d.edge_index_1.shape == (2, 1)
    assert d.edge_index_2.shape == (2, 0)    # empty relation still present
    assert d.lvl.item() == 2
    assert d.num_nodes == 3


def test_collate_offsets_multi_relation_edges():
    v = Vocab.build([["a", "b"]], 8)
    d1 = view_to_data(["a", "b"], [(0, 1, 0)], v, lvl=0)
    d2 = view_to_data(["a", "b"], [(0, 1, 0)], v, lvl=1)
    b = collate_programs([[d1], [d2]])       # 2 programs, 1 view each
    ei = b.edge_index_0
    assert ei.shape == (2, 2)
    # second graph's edge (0,1) must be offset by its 2 nodes -> (2,3)
    assert set(ei[0].tolist()) == {0, 2}
    assert set(ei[1].tolist()) == {1, 3}
    assert b.prog.tolist() == [0, 1]
    assert b.lvl.tolist() == [0, 1]


def test_collate_program_and_level_labels():
    v = Vocab.build([["a", "b", "c"]], 16)
    prog0 = [view_to_data(["a", "b"], [(0, 1, 0)], v, l) for l in range(4)]
    prog1 = [view_to_data(["a", "c"], [(0, 1, 1)], v, l) for l in range(4)]
    b = collate_programs([prog0, prog1])     # 2 programs x 4 views
    assert b.prog.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert b.lvl.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
