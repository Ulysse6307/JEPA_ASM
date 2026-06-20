"""Compile C/C++ source to textual LLVM IR (.ll) via clang.

We emit *textual* IR (`-emit-llvm -S`) because the graph builder parses IR text
with llvmlite (pure Python, no native LLVM build required).

Compilation flags are chosen so that single functions from corpora like
AnghaBench compile in isolation:
  -O1            : enough optimization to get clean SSA + mem2reg, without the
                   aggressive inlining/vectorization of -O2/-O3 that would distort
                   the graph. (-O0 leaves everything in memory via alloca/load/store
                   and hides data flow behind the stack — bad for our data-flow edges.)
  -g0            : no debug info (keeps IR small)
  -Xclang -disable-llvm-passes is intentionally NOT used: we want mem2reg to run so
                   SSA def-use edges are meaningful.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class IRCompileError(RuntimeError):
    """Raised when clang fails to produce IR for a source unit."""


def find_clang() -> str:
    """Locate a clang binary. Honors $CLANG, then PATH.

    On macOS the Apple clang shipped with Xcode works fine for -emit-llvm.
    """
    env = os.environ.get("CLANG")
    if env:
        return env
    for name in ("clang", "clang-18", "clang-17", "clang-16", "clang-15"):
        found = shutil.which(name)
        if found:
            return found
    raise IRCompileError(
        "No clang found. Install it (macOS: `xcode-select --install`) or set $CLANG."
    )


# Optimization level chosen to expose SSA data flow without distorting structure.
# -O1 is the default used to BUILD the training corpus; other levels can be passed
# explicitly (e.g. to probe O0-vs-O3 robustness of the learned representation).
_DEFAULT_OPT = "-O1"
_BASE_FLAGS = ("-g0", "-fno-discard-value-names")


def _lang_flags(path: Path) -> tuple[str, ...]:
    if path.suffix in (".cpp", ".cc", ".cxx", ".C", ".hpp"):
        return ("-x", "c++", "-std=c++17")
    return ("-x", "c", "-std=c11")


def compile_to_ir(
    source_path: str | os.PathLike,
    out_path: str | os.PathLike | None = None,
    *,
    opt_level: str = _DEFAULT_OPT,
    extra_flags: tuple[str, ...] = (),
    timeout: float = 60.0,
) -> str:
    """Compile a C/C++ source file to textual LLVM IR.

    ``opt_level`` is the clang -O flag ("-O0".."-O3"); it REPLACES the default,
    so callers can probe e.g. O0 vs O3. Returns the IR as a string. If
    ``out_path`` is given, also writes it there. Raises IRCompileError on failure.
    """
    src = Path(source_path)
    if not src.is_file():
        raise IRCompileError(f"source not found: {src}")
    clang = find_clang()

    # We write IR to stdout via `-o -` to avoid temp files for the common path.
    cmd = [
        clang, "-emit-llvm", "-S",
        opt_level, *_BASE_FLAGS, *_lang_flags(src), *extra_flags,
        str(src), "-o", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise IRCompileError(f"clang timed out on {src}") from e

    if proc.returncode != 0:
        raise IRCompileError(
            f"clang failed on {src} (exit {proc.returncode}):\n{proc.stderr.strip()[:2000]}"
        )
    ir = proc.stdout
    if not ir.strip():
        raise IRCompileError(f"clang produced empty IR for {src}")

    if out_path is not None:
        Path(out_path).write_text(ir)
    return ir


def compile_source_string(
    source: str,
    *,
    is_cpp: bool = False,
    opt_level: str = _DEFAULT_OPT,
    extra_flags: tuple[str, ...] = (),
    timeout: float = 60.0,
) -> str:
    """Compile a C/C++ source given as a string. Useful for tests and corpora
    that hold source inline (e.g. JSON datasets) rather than as files.
    """
    suffix = ".cpp" if is_cpp else ".c"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(source)
        tmp = f.name
    try:
        return compile_to_ir(tmp, opt_level=opt_level, extra_flags=extra_flags,
                             timeout=timeout)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass