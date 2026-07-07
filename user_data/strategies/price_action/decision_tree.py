"""Binary decision tree loader (minimal port from PA_Agent).

Parses ``二元决策.txt`` into a ``{node_id: question}`` mapping so the
trace normalize layer can repair AI-written node questions to match
the canonical wording. Only the parsing surface needed by
``canonical_tree_questions`` is ported; the full upstream module has
trace merging, validators, and label helpers that remain unported.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


_BINARY_DECISION_FILE = "二元决策.txt"
_PROMPT_DIR = Path(__file__).with_name("prompt_templates")

_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$")
_NODE_RE = re.compile(r"^###\s+([\d.]+[A-Z]?)\s+(.+)$")


@lru_cache(maxsize=1)
def load_decision_tree(path: Path | None = None) -> dict[str, Any]:
    """Parse ``二元决策.txt`` into ``{sections, node_index}``.

    Mirrors upstream ``load_decision_tree`` but skips the
    branch-outcome extraction (not needed for question canonicalization).
    Result is cached via :func:`functools.lru_cache`.
    """
    txt_path = path or (_PROMPT_DIR / _BINARY_DECISION_FILE)
    text = txt_path.read_text(encoding="utf-8")

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in text.splitlines():
        sec_m = _SECTION_RE.match(line)
        if sec_m:
            if current is not None:
                sections.append(current)
            current = {
                "id": sec_m.group(1),
                "title": sec_m.group(2).strip(),
                "nodes": [],
            }
            continue

        node_m = _NODE_RE.match(line)
        if node_m and current is not None:
            current["nodes"].append(
                {
                    "id": node_m.group(1),
                    "question": node_m.group(2).strip(),
                }
            )

    if current is not None:
        sections.append(current)

    node_index: dict[str, dict[str, Any]] = {}
    for sec in sections:
        for node in sec["nodes"]:
            nid = node["id"]
            node_index[nid] = {
                **node,
                "section_id": sec["id"],
                "section_title": sec["title"],
            }

    return {
        "version": 1,
        "source": txt_path.name,
        "sections": sections,
        "node_index": node_index,
    }


def canonical_tree_questions() -> dict[str, str]:
    """Return ``{node_id: canonical_question}`` for non-empty questions.

    Mirrors upstream ``_canonical_tree_questions``. Used by trace
    normalize to overwrite AI-paraphrased questions with the canonical
    wording from the decision tree spec.
    """
    tree = load_decision_tree()
    index = tree.get("node_index", {})
    return {
        str(nid): str(node.get("question", "") or "").strip()
        for nid, node in index.items()
        if str(node.get("question", "") or "").strip()
    }
