"""End-to-end smoke tests for the JEPA-IR pipeline.

Covers: source -> IR -> 3-relation graph -> PyG data -> masking -> model
forward/backward -> VICReg loss. Skips gracefully if clang is unavailable.
"""
from __future__ import annotations

import shutil

import pytest
import torch

from jepa_ir.config import Config, MaskConfig, ModelConfig, VICRegConfig
from jepa_ir.data import collate_views, mask_graph, program_graph_to_data
from jepa_ir.graph import build_graph_from_ir, build_graph_from_source
from jepa_ir.model import JEPAModel
from jepa_ir.train.vicreg import vicreg_loss

clang_missing = shutil.which("clang") is None and "CLANG" not in __import__("os").environ
needs_clang = pytest.mark.skipif(clang_missing, reason="clang not available")

SUM_ARRAY = """
int sum_array(const int *a, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) sum += a[i];
    return sum;
}
"""

# A pre-compiled IR snippet so graph tests run even without clang.
RAW_IR = """
define i32 @add2(i32 %x, i32 %y) {
entry:
  %s = add i32 %x, %y
  %t = mul i32 %s, %s
  ret i32 %t
}
"""


def test_graph_from_raw_ir_has_three_relations():
    graphs = build_graph_from_ir(RAW_IR)
    assert len(graphs) == 1
    g = graphs[0]
    assert g.num_nodes == 3                     # add, mul, ret
    # data flow: %s -> mul, %s -> mul (twice), %t -> ret
    assert g.num_edges("data") >= 2
    # all three relation buckets exist
    assert set(g.edges.keys()) == {"control", "data", "memory"}


@needs_clang
def test_source_to_graph_all_relations():
    g = build_graph_from_source(SUM_ARRAY)[0]
    assert g.num_nodes > 5
    assert g.num_edges("control") > 0           # the loop has branches
    assert g.num_edges("data") > 0              # SSA def-use


@needs_clang
def test_masking_preserves_topology_and_marks_holes():
    data = program_graph_to_data(build_graph_from_source(SUM_ARRAY)[0])
    view = mask_graph(data, MaskConfig(mask_ratio=0.3), torch.Generator().manual_seed(0))
    # same node count in both views
    assert view.context.num_nodes == view.target.num_nodes == data.num_nodes
    # at least one node masked, not all
    assert 0 < int(view.mask.sum()) < data.num_nodes
    # masked context nodes carry the sentinel opcode 0
    assert (view.context.x_opcode[view.mask] == 0).all()
    # target keeps full opcodes
    assert (view.target.x_opcode == data.x_opcode).all()


def test_model_forward_backward_and_vicreg():
    # build two graphs from raw IR (no clang needed)
    g1 = build_graph_from_ir(RAW_IR)[0]
    g2 = build_graph_from_ir(RAW_IR.replace("add2", "add3").replace("mul", "add"))[0]
    views = [
        mask_graph(program_graph_to_data(g), MaskConfig(mask_ratio=0.5, min_kept_nodes=1),
                   torch.Generator().manual_seed(i))
        for i, g in enumerate((g1, g2))
    ]
    ctx, tgt = collate_views(views)

    cfg = ModelConfig(hidden_dim=32, embedding_dim=48, num_layers=2)
    model = JEPAModel(cfg)
    # forward returns the latent embeddings directly (no projector, eb_jepa-style)
    z_a, z_b = model(ctx, tgt)
    assert z_a.shape == z_b.shape == (2, cfg.embedding_dim)

    out = vicreg_loss(z_a, z_b, VICRegConfig())
    assert torch.isfinite(out.total)
    out.total.backward()
    # the mask token must receive gradient (masked-node path is exercised)
    assert model.encoder.mask_token.grad is not None

    # encode() is the deliverable API: graph -> embedding (same as forward branch)
    emb = model.encode(tgt)
    assert emb.shape == (2, cfg.embedding_dim)


def test_vicreg_penalizes_collapse():
    # two identical constant batches: invariance ~0 but variance term should be high
    z = torch.zeros(8, 16)
    out = vicreg_loss(z, z, VICRegConfig())
    # variance hinge is maximal (std=0 -> relu(1-0)=1 per dim) -> var term ~2.0
    assert out.variance > 1.0
    assert out.invariance < 1e-6