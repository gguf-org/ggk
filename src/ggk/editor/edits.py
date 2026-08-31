"""Apply a GUI edit spec to a parsed GGUF file.

The edit spec is the JSON the web GUI accumulates (mirroring the state model
of the chrome-extension/desktop editors):

    {
      "metadata_edits":    {key: "edited display string"},
      "deleted_meta_keys": [key, ...],
      "new_meta_rows":     [{"key":.., "value":.., "type": int}, ...],
      "tensor_renames":    {"<orig index>": "new name"},
      "deleted_tensors":   [index, ...],
      "merges":            [{"name":.., "dtype":.., "shape":[..], "parts":[indices]}],
      "added_tensors":     [{"name":.., "shape":[..], "dtype":..,
                             "source": {"kind":"zeros"} |
                                       {"kind":"external","path":..,"offset":..,"size":..}}],
      "order":             ["t:<idx>"|"m:<pos>"|"a:<pos>", ...] | null
    }

Row keys use list *positions* for merges ("m:0" = merges[0]) and added
tensors ("a:0" = added_tensors[0]).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .gguf import (
    ParsedGGUF,
    TensorInfo,
    WritePlan,
    build_header,
    build_updated_metadata,
    tensor_byte_size,
)


def _default_order(parsed: ParsedGGUF, edits: Dict[str, Any]) -> List[str]:
    return (
        [f"t:{i}" for i in range(len(parsed.tensor_infos))]
        + [f"m:{i}" for i in range(len(edits.get("merges", [])))]
        + [f"a:{i}" for i in range(len(edits.get("added_tensors", [])))]
    )


def _ordered_keys(parsed: ParsedGGUF, edits: Dict[str, Any]) -> List[str]:
    default = _default_order(parsed, edits)
    order = edits.get("order")
    if not order:
        return default
    valid = set(default)
    kept = [k for k in order if k in valid]
    kept_set = set(kept)
    return kept + [k for k in default if k not in kept_set]


def compute_final_tensors(
    parsed: ParsedGGUF, edits: Dict[str, Any]
) -> Tuple[List[TensorInfo], List[WritePlan]]:
    """Final header tensor list (offsets recomputed) + per-tensor write plans."""
    alignment = parsed.alignment
    renames = {int(k): v for k, v in edits.get("tensor_renames", {}).items()}
    deleted = set(int(i) for i in edits.get("deleted_tensors", []))
    merges = edits.get("merges", [])
    added = edits.get("added_tensors", [])
    merged_source = {int(i) for m in merges for i in m["parts"]}

    finals: List[TensorInfo] = []
    plans: List[WritePlan] = []
    running = 0

    def push(name: str, shape: List[int], dtype: int, plan: WritePlan):
        nonlocal running
        finals.append(TensorInfo(name, list(shape), int(dtype), running))
        plans.append(plan)
        running += -(-plan.size // alignment) * alignment

    for key in _ordered_keys(parsed, edits):
        kind, _, idx_s = key.partition(":")
        idx = int(idx_s)

        if kind == "t":
            tensor = parsed.tensor_infos[idx]
            if idx in deleted or idx in merged_source:
                continue
            size = parsed.tensor_sizes[idx]
            push(renames.get(idx, tensor.name), tensor.shape, tensor.dtype, WritePlan(
                size=size,
                segments=[{
                    "path": parsed.path,
                    "offset": parsed.tensor_data_offset + tensor.offset,
                    "length": size,
                }],
            ))
        elif kind == "m":
            merge = merges[idx]
            segments = []
            size = 0
            for part in merge["parts"]:
                tensor = parsed.tensor_infos[int(part)]
                part_size = tensor_byte_size(tensor.dtype, tensor.shape)
                segments.append({
                    "path": parsed.path,
                    "offset": parsed.tensor_data_offset + tensor.offset,
                    "length": part_size,
                })
                size += part_size
            push(merge["name"], merge["shape"], merge["dtype"],
                 WritePlan(size=size, segments=segments))
        else:  # "a"
            add = added[idx]
            source = add.get("source", {})
            if source.get("kind") == "external":
                size = int(source["size"])
                plan = WritePlan(size=size, segments=[{
                    "path": source["path"],
                    "offset": int(source["offset"]),
                    "length": size,
                }])
            else:
                size = tensor_byte_size(int(add["dtype"]), list(add["shape"]))
                plan = WritePlan(size=size, zero_fill=size)
            push(add["name"], add["shape"], add["dtype"], plan)

    names = set()
    for t in finals:
        if not t.name.strip():
            raise ValueError("Every tensor needs a non-empty name.")
        if t.name in names:
            raise ValueError(f'Duplicate tensor name "{t.name}" — rename before saving.')
        names.add(t.name)

    return finals, plans


def build_edited_header(parsed: ParsedGGUF, edits: Dict[str, Any],
                        finals: List[TensorInfo]) -> bytes:
    metadata = build_updated_metadata(
        parsed.metadata,
        edits.get("metadata_edits", {}),
        set(edits.get("deleted_meta_keys", [])),
        edits.get("new_meta_rows", []),
    )
    return build_header(parsed.version, metadata, finals, parsed.alignment)
