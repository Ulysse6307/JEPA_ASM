"""Build the lossless 3-relation graph from textual LLVM IR.

We use llvmlite to *parse and verify* the module (catching malformed IR early),
then walk the textual IR to extract, per function:

  nodes   : one per instruction
  control : CFG edges between basic blocks, lifted onto instructions
            (terminator of block B -> first instruction of each successor block)
  data    : SSA def-use edges (the node defining %x -> every node that lists %x
            as an operand)
  memory  : program-order chain among memory-touching instructions within a block
            (load/store/atomic/call/fence...), encoding side-effect ordering

Why parse text rather than use llvmlite's value API: llvmlite's binding layer
does not expose a rich operand graph from Python (that lives in the C++ API). The
textual form is stable and complete, so we parse it directly. llvmlite still does
the heavy lifting of validating the IR for us.

Limitations (documented, not hidden):
  * Operand extraction is name-based (%name / @global). Inline constants are not
    nodes (correct: they carry no def). Pointer aliasing is not resolved, so the
    memory relation is a conservative *program order*, not a precise dependence.
    That matches the project's "side-effect ordering" requirement, which is about
    order, not alias-precise dependence.
"""
from __future__ import annotations

import re

from ..config import OPCODE_TO_ID
from .schema import Node, ProgramGraph


class GraphBuildError(RuntimeError):
    pass


def _parse_and_verify(ir_text: str):
    """Parse + verify IR with llvmlite, tolerant of API changes across versions.

    llvmlite >=0.44 auto-initializes LLVM and *removed* the old initialize()
    helpers (calling them now raises). Older versions require them. We try the
    modern direct path first and only fall back to explicit init on the specific
    "not initialized" failure.
    """
    import llvmlite.binding as llvm

    try:
        mod = llvm.parse_assembly(ir_text)
    except RuntimeError as e:
        msg = str(e).lower()
        if "initialize" in msg and hasattr(llvm, "initialize"):
            llvm.initialize()
            llvm.initialize_native_target()
            llvm.initialize_native_asmprinter()
            mod = llvm.parse_assembly(ir_text)
        else:
            raise
    mod.verify()
    return mod


# Opcodes that touch memory / have observable side effects -> participate in the
# memory-ordering relation.
_MEMORY_OPCODES = {
    "load", "store", "atomicrmw", "cmpxchg", "fence",
    "call", "invoke",          # calls may read/write memory (conservative)
}
_TERMINATOR_OPCODES = {
    "ret", "br", "switch", "indirectbr", "invoke", "resume",
    "unreachable", "cleanupret", "catchret", "catchswitch",
}

# --- regexes for the textual IR ------------------------------------------- #

# A value-defining instruction:  %name = opcode ...
_DEF_RE = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*(.+)$")
# Leading opcode word of an instruction body (after optional "tail", "fast-math"
# flags etc. we just grab the first identifier-looking token).
_OPCODE_RE = re.compile(r"^([a-zA-Z_][\w.]*)")
# Operand value references: SSA locals (%x) and globals (@g). We capture both.
_VALUE_REF_RE = re.compile(r"(?<![\w.])([%@][\w.\-]+)")
# Basic-block label line:  "label:" or "; <label>:N" style. llvm prints
# "<name>:" at column 0 (possibly with a preds comment after).
_LABEL_RE = re.compile(r"^([\w.\-]+):")
# Function definition start / end.
_FUNC_DEF_RE = re.compile(r"^define\s+.*?@([\w.\-$]+)\s*\(")


def _strip_comment(line: str) -> str:
    # Remove trailing "; ..." comments but keep ones that are inside strings? IR
    # rarely embeds ';' in strings at instruction level; this is good enough.
    idx = line.find(";")
    return line[:idx] if idx != -1 else line


def _first_opcode(body: str) -> str:
    # body may start with flags like "tail call", "nsw add", "fast fadd",
    # "atomic load" ... We strip a known set of leading modifiers, then take the
    # first token as the opcode.
    modifiers = {
        "tail", "musttail", "notail", "nsw", "nuw", "exact", "fast",
        "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc",
        "atomic", "volatile", "weak", "acquire", "release", "acq_rel",
        "seq_cst", "monotonic", "unordered", "inbounds", "synccope",
    }
    tokens = body.replace(",", " ").split()
    for tok in tokens:
        m = _OPCODE_RE.match(tok)
        if not m:
            continue
        word = m.group(1)
        if word in modifiers:
            continue
        return word
    return "<unk>"


def _split_functions(ir_text: str) -> list[tuple[str, list[str]]]:
    """Return [(func_name, body_lines)] for every `define ... { ... }`."""
    funcs: list[tuple[str, list[str]]] = []
    lines = ir_text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = _FUNC_DEF_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        name = m.group(1)
        # find the body between the first '{' and the matching '}'
        depth = 0
        body: list[str] = []
        started = False
        while i < n:
            line = lines[i]
            depth += line.count("{") - line.count("}")
            if "{" in line and not started:
                started = True
                i += 1
                continue
            if started and depth <= 0:
                break
            if started:
                body.append(line)
            i += 1
        funcs.append((name, body))
        i += 1
    return funcs


def _build_one_function(name: str, body: list[str]) -> ProgramGraph:
    g = ProgramGraph(name=name)

    # First pass: collect instructions, assign node indices, track blocks, and
    # remember which block each label starts and the def-site of each SSA name.
    block_idx = -1
    block_first_node: dict[int, int] = {}     # block -> first node idx
    block_term_node: dict[int, int] = {}      # block -> terminator node idx
    label_to_block: dict[str, int] = {}
    node_block: list[int] = []
    def_site: dict[str, int] = {}             # ssa name -> node idx that defines it
    # successor labels recorded from terminators, resolved in a later pass
    block_successors: dict[int, list[str]] = {}

    # entry block (instructions before the first explicit label) is block 0
    pending_entry = True

    for raw in body:
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue

        # Block label?
        lab = _LABEL_RE.match(line.strip())
        # A label line has no '=' and ends the previous block. llvm label lines
        # look like "if.then:" optionally followed by spaces.
        is_label = bool(lab) and "=" not in line and not line.strip().startswith("define")
        if is_label:
            block_idx += 1
            label_to_block[lab.group(1)] = block_idx
            pending_entry = False
            continue

        # If we have instructions before any label, open the implicit entry block.
        if pending_entry:
            block_idx = 0
            pending_entry = False
            # entry block has no textual label; give it a synthetic key
            label_to_block.setdefault("%__entry__", 0)

        instr = line.strip()
        # Defining instruction?  %x = body
        dm = _DEF_RE.match(instr)
        if dm:
            result_name = dm.group(1)
            opbody = dm.group(2)
        else:
            result_name = None
            opbody = instr

        opcode = _first_opcode(opbody)
        idx = len(g.nodes)

        # operands: all value refs in the body, minus the defined name itself
        operand_refs = _VALUE_REF_RE.findall(opbody)
        operands = [o for o in operand_refs if o != result_name]

        is_term = opcode in _TERMINATOR_OPCODES
        is_mem = opcode in _MEMORY_OPCODES
        produces = result_name is not None

        node = Node(
            idx=idx,
            opcode=opcode,
            block=block_idx,
            result_name=result_name,
            operands=operands,
            is_terminator=is_term,
            is_memory_op=is_mem,
            produces_value=produces,
            text=instr,
        )
        g.nodes.append(node)
        node_block.append(block_idx)

        if block_idx not in block_first_node:
            block_first_node[block_idx] = idx
        if result_name is not None:
            def_site[result_name] = idx

        # Record successors from terminators (branch targets are %labels).
        if is_term:
            block_term_node[block_idx] = idx
            # branch/switch targets appear as label operands; in textual IR they
            # look like "label %if.then". Capture %names that match known labels
            # later (resolved after we have all labels).
            targets = re.findall(r"label\s+(%[\w.\-]+)", opbody)
            # switch default + cases also use "label %x"
            block_successors[block_idx] = [t[1:] for t in targets]  # drop '%'

    # ---- DATA edges: def -> use ---------------------------------------- #
    for node in g.nodes:
        for op in node.operands:
            src = def_site.get(op)
            if src is not None and src != node.idx:
                g.add_edge("data", src, node.idx)

    # ---- CONTROL edges: terminator(B) -> first instr of successor ------- #
    # label names captured without '%'; map them to blocks
    name_to_block = {k.lstrip("%"): v for k, v in label_to_block.items()}
    for b, succ_labels in block_successors.items():
        term = block_term_node.get(b)
        if term is None:
            continue
        for lab in succ_labels:
            sb = name_to_block.get(lab)
            if sb is None:
                continue
            tgt = block_first_node.get(sb)
            if tgt is not None:
                g.add_edge("control", term, tgt)

    # ---- MEMORY edges: program order among memory ops --------------------#
    # Conservative side-effect ordering. We chain memory-touching instructions
    # in program order. Within a block the order is exact (textual order). Across
    # blocks we additionally link the last memory op of a block to the first
    # memory op of each CFG-successor block, so the ordering follows control flow
    # rather than asserting a spurious total order over unordered blocks.
    mem_in_block: dict[int, list[int]] = {}
    for node in g.nodes:
        if node.is_memory_op:
            mem_in_block.setdefault(node.block, []).append(node.idx)
    for nodes_ in mem_in_block.values():
        nodes_.sort()
        for a, c in zip(nodes_, nodes_[1:]):
            g.add_edge("memory", a, c)
    # cross-block: last mem op of B -> first mem op of each successor S
    for b, succ_labels in block_successors.items():
        if b not in mem_in_block:
            continue
        last = mem_in_block[b][-1]
        for lab in succ_labels:
            sb = name_to_block.get(lab)
            if sb is not None and sb in mem_in_block:
                g.add_edge("memory", last, mem_in_block[sb][0])

    return g


def build_graph_from_ir(ir_text: str, *, verify: bool = True) -> list[ProgramGraph]:
    """Parse textual LLVM IR and return one ProgramGraph per defined function.

    If ``verify`` is True, llvmlite parses+verifies the module first and raises
    GraphBuildError on malformed IR.
    """
    if verify:
        try:
            mod = _parse_and_verify(ir_text)
            del mod
        except ImportError:
            # llvmlite not installed: skip verification, still build from text.
            pass
        except RuntimeError as e:
            raise GraphBuildError(f"LLVM failed to verify IR: {e}") from e

    graphs: list[ProgramGraph] = []
    for name, body in _split_functions(ir_text):
        if not body:
            continue  # declaration only, no body
        g = _build_one_function(name, body)
        if g.num_nodes > 0:
            graphs.append(g)
    if not graphs:
        raise GraphBuildError("no function bodies found in IR")
    return graphs


def build_graph_from_source(source: str, *, is_cpp: bool = False) -> list[ProgramGraph]:
    """Convenience: C/C++ source string -> graphs (compiles, then builds)."""
    from ..ir import compile_source_string

    ir_text = compile_source_string(source, is_cpp=is_cpp)
    return build_graph_from_ir(ir_text)


# Re-export for callers that want the opcode id mapping without importing config.
def opcode_id(opcode: str) -> int:
    from ..config import OPCODE_UNK

    return OPCODE_TO_ID.get(opcode, OPCODE_UNK)