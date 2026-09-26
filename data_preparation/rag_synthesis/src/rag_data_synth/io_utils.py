from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

JsonDict = Dict[str, Any]


def read_jsonl(path: str | Path, *, limit: int = 0) -> List[JsonDict]:
    path = Path(path)
    out: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
                if limit and len(out) >= limit:
                    break
    return out


def iter_jsonl(path: str | Path, *, limit: int = 0) -> Iterator[JsonDict]:
    path = Path(path)
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
                n += 1
                if limit and n >= limit:
                    return


def write_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json_or_empty(path: str | Path | None) -> JsonDict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {p}, got {type(obj)}")
    return obj
