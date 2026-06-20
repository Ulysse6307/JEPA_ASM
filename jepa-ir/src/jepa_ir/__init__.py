"""JEPA-IR: self-supervised GNN encoder for program representations.

Pipeline:  code -> LLVM IR -> lossless 3-relation graph -> GNN encoder -> embedding

The graph keeps three relations at once (never discarded):
  1. control flow      (branch / fallthrough between basic blocks)
  2. data flow         (def-use edges between SSA values)
  3. memory ordering   (side-effect order among memory ops in a block)

Training is JEPA-style (no decoder): encode a masked graph and the full graph,
then pull their embeddings together in latent space, with VICReg regularization
to forbid collapse.
"""

__version__ = "0.1.0"