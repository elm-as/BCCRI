"""Automated ELMAS.md structural, complexity, and dependency-DAG tests."""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
FORBIDDEN_NAMES = frozenset({"utils", "helpers", "misc", "common", "temp", "final", "v2"})
ALLOWED_DAG_DEPS = {
    "core": set(),
    "ingestion": {"core"},
    "features": {"core"},
    "modeling": {"core", "features"},
    "evaluation": {"core", "features", "modeling"},
    "visualization": {"core", "evaluation", "modeling"},
}
COMPOUND_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
)


def _max_block_depth(node: ast.AST, depth: int = 0) -> int:
    """Compute maximum nested control-flow block depth inside an AST node."""
    best = depth
    for child in ast.iter_child_nodes(node):
        next_depth = depth + 1 if isinstance(child, COMPOUND_NODES) else depth
        best = max(best, _max_block_depth(child, next_depth))
    return best


def test_elmas_file_length_indent_and_parameter_limits() -> None:
    """Every src/*.py file obeys <=350 lines, <=3 indent levels, and <=3 params/function."""
    py_files = sorted(SRC_ROOT.rglob("*.py"))
    assert len(py_files) >= 8
    for py_path in py_files:
        assert not any(part in FORBIDDEN_NAMES for part in py_path.parts), f"Forbidden path: {py_path}"
        text = py_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) <= 350, f"{py_path.name} exceeds 350 lines ({len(lines)})"
        tree = ast.parse(text, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                param_count = len(node.args.posonlyargs) + len(node.args.args) + len(node.args.kwonlyargs)
                assert param_count <= 3, f"{py_path.name}:{node.name} has {param_count} params"
                depth = _max_block_depth(node)
                assert depth <= 3, f"{py_path.name}:{node.name} has nesting depth {depth}"


def test_elmas_unidirectional_domain_dag() -> None:
    """Domain packages strictly follow visualization->evaluation->modeling->features->ingestion->core."""
    for py_path in sorted(SRC_ROOT.rglob("*.py")):
        rel = py_path.relative_to(SRC_ROOT)
        if len(rel.parts) < 2:
            continue
        pkg = rel.parts[0]
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                target_pkg = node.module.split(".")[1]
                if target_pkg != pkg:
                    assert target_pkg in ALLOWED_DAG_DEPS[pkg], (
                        f"DAG violation: {pkg} imports {target_pkg} in {py_path.name}"
                    )
