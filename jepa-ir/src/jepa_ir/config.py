"""Central configuration: graph schema + all hyperparameters.

Single source of truth. The graph schema (edge types, node feature layout) is
shared between the graph builder, the dataset, and the model, so it lives here
to avoid drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Graph schema                                                                 #
# --------------------------------------------------------------------------- #

# The three relations of the lossless IR graph. Order is fixed: the model builds
# one message-passing stack per edge type and relies on this ordering.
EDGE_TYPES: tuple[str, ...] = ("control", "data", "memory")

# Node = one LLVM instruction. We give each node a small, fixed feature vector
# rather than learning a giant opcode embedding table up front. The opcode is the
# main signal; the rest are cheap structural/type hints.
#
# A node feature vector is:  [opcode_id]  (categorical, embedded in the model)
#                          +  [structural floats]  (see STRUCT_FEATURES)
#
# We keep opcode as an *index* (embedded in the model) and the structural part as
# raw floats concatenated after the opcode embedding.

# Curated opcode vocabulary. Unknown opcodes map to OPCODE_UNK. Kept compact on
# purpose; extend as the corpus demands. Index 0 is reserved for the mask token's
# placeholder opcode so a masked node never collides with a real opcode.
OPCODE_VOCAB: tuple[str, ...] = (
    "<pad>",        # 0 - reserved (also used as masked-node opcode placeholder)
    "<unk>",        # 1 - opcode not in vocab
    # memory
    "load", "store", "alloca", "getelementptr", "fence", "atomicrmw", "cmpxchg",
    # arithmetic / logic
    "add", "sub", "mul", "udiv", "sdiv", "urem", "srem",
    "fadd", "fsub", "fmul", "fdiv", "frem",
    "shl", "lshr", "ashr", "and", "or", "xor",
    # compare
    "icmp", "fcmp",
    # control / terminators
    "br", "switch", "ret", "indirectbr", "unreachable",
    # calls
    "call", "invoke",
    # casts
    "trunc", "zext", "sext", "fptrunc", "fpext",
    "fptoui", "fptosi", "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast",
    # aggregate / ssa
    "phi", "select", "extractvalue", "insertvalue", "extractelement", "insertelement",
)
OPCODE_PAD = 0
OPCODE_UNK = 1
OPCODE_TO_ID: dict[str, int] = {op: i for i, op in enumerate(OPCODE_VOCAB)}

# Structural float features appended after the opcode embedding (kept tiny).
#   is_terminator, is_memory_op, num_operands (normalized), is_in_loop_header(0/1 placeholder)
STRUCT_FEATURES: tuple[str, ...] = (
    "is_terminator",
    "is_memory_op",
    "num_operands_norm",
    "produces_value",
)
NUM_STRUCT_FEATURES = len(STRUCT_FEATURES)


# --------------------------------------------------------------------------- #
# Hyperparameters                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    opcode_vocab_size: int = len(OPCODE_VOCAB)
    opcode_embed_dim: int = 64
    num_struct_features: int = NUM_STRUCT_FEATURES
    hidden_dim: int = 128
    num_layers: int = 4          # message-passing rounds
    edge_types: tuple[str, ...] = EDGE_TYPES
    dropout: float = 0.1
    # final program embedding (the deliverable). VICReg acts on this directly —
    # no projector head (eb_jepa convention).
    embedding_dim: int = 128


@dataclass
class MaskConfig:
    # Fraction of nodes to mask in the context (masked) view.
    mask_ratio: float = 0.30
    # Mask whole basic blocks instead of random nodes (more structural difficulty).
    block_masking: bool = True
    min_kept_nodes: int = 4      # never mask a graph down below this
    # Also drop edges INCIDENT to masked nodes (a real structural hole, not just
    # greyed-out content). With this off, the GNN can reconstruct masked nodes
    # from intact topology, so the task is too easy. On = the harder JEPA task.
    mask_edges: bool = False


@dataclass
class VICRegConfig:
    # Weights of the three VICReg terms. Invariance pulls views together;
    # variance + covariance forbid collapse (mandatory per project spec).
    # Coefficients aligned on the eb_jepa reference repo (sim/std/cov = 1/1/1),
    # which regularizes the latent directly (no projector) — rather than the
    # 25/25/1 of the original (image) VICReg paper.
    sim_coeff: float = 1.0       # invariance (MSE between the two views)
    std_coeff: float = 1.0       # variance hinge
    cov_coeff: float = 1.0       # covariance off-diagonal penalty
    eps: float = 1e-4


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    num_workers: int = 0
    device: str = "auto"         # "auto" -> cuda if available else cpu
    seed: int = 0
    log_every: int = 20
    ckpt_dir: str = "checkpoints"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    vicreg: VICRegConfig = field(default_factory=VICRegConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def default_config() -> Config:
    return Config()