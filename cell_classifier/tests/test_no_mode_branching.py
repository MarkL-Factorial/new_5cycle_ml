"""AST audit: dispatch-style `if mode == "…"` and `if model_name == "…"`
belong only in the designated dispatch sites.

Walks every .py file under src/cell_classifier/. Only flags equality
comparisons (==) of a `mode` / `model_name` identifier (or `config["mode"]`
subscript form) against a string literal. Validators like
`if mode not in (...)` are NOT flagged — they don't dispatch behavior, they
guard inputs.
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src" / "cell_classifier"

_ALLOWED_MODE_BRANCH_FILES = {
    Path("cli.py"),
    Path("data") / "splits.py",
}
_ALLOWED_MODEL_BRANCH_FILES = {
    Path("models") / "registry.py",
}

_MODE_NAMES = {"mode"}
_MODEL_NAMES = {"model_name", "model"}


def _expr_is_target(expr: ast.AST, name_set: set[str]) -> bool:
    """True if `expr` is a bare Name in name_set or a Subscript on such a name
    with a string literal index (e.g., config["mode"]).
    """
    if isinstance(expr, ast.Name) and expr.id in name_set:
        return True
    if isinstance(expr, ast.Subscript):
        # config["mode"] / args.mode (Attribute) etc.
        sub = expr.slice
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in name_set:
            return True
    if isinstance(expr, ast.Attribute) and expr.attr in name_set:
        return True
    return False


def _has_eq_dispatch(node: ast.If, name_set: set[str]) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not test.ops or not all(isinstance(op, ast.Eq) for op in test.ops):
        return False
    # Either side may be the identifier; the other must be a string literal
    candidates = [test.left, *test.comparators]
    has_target = any(_expr_is_target(c, name_set) for c in candidates)
    has_string = any(
        isinstance(c, ast.Constant) and isinstance(c.value, str) for c in candidates
    )
    return has_target and has_string


def _scan(py_path: Path) -> tuple[bool, bool]:
    """Return (uses_mode_branch, uses_model_branch)."""
    tree = ast.parse(py_path.read_text())
    uses_mode, uses_model = False, False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if _has_eq_dispatch(node, _MODE_NAMES):
            uses_mode = True
        if _has_eq_dispatch(node, _MODEL_NAMES):
            uses_model = True
    return uses_mode, uses_model


def test_no_mode_branching_outside_allowed():
    offenders: list[Path] = []
    for py in _SRC.rglob("*.py"):
        rel = py.relative_to(_SRC)
        uses_mode, _ = _scan(py)
        if uses_mode and rel not in _ALLOWED_MODE_BRANCH_FILES:
            offenders.append(rel)
    assert offenders == [], (
        f"`if mode == …` outside {_ALLOWED_MODE_BRANCH_FILES}: {offenders}"
    )


def test_no_model_name_branching_outside_registry():
    offenders: list[Path] = []
    for py in _SRC.rglob("*.py"):
        rel = py.relative_to(_SRC)
        _, uses_model = _scan(py)
        if uses_model and rel not in _ALLOWED_MODEL_BRANCH_FILES:
            offenders.append(rel)
    assert offenders == [], (
        f"`if model_name == …` outside {_ALLOWED_MODEL_BRANCH_FILES}: {offenders}"
    )
