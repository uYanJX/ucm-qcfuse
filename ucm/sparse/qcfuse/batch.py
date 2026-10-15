"""Per-step request layout for batched QCFuse prefill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from ucm.sparse.qcfuse.blend_info import BlendStyle, QcFuseRequestState


@dataclass
class _AttentionLayout:
    tensors: tuple
    producer: Optional[torch.cuda.Stream] = None
    ready: Optional[torch.cuda.Event] = None

    def consume(self):
        if self.producer is not None:
            current = torch.cuda.current_stream(self.tensors[0].device)
            if current.cuda_stream != self.producer.cuda_stream:
                current.wait_event(self.ready)
            for tensor in self.tensors:
                tensor.record_stream(current)
        return self.tensors


@dataclass(frozen=True)
class QcFuseStepEntry:
    request_id: str
    state: QcFuseRequestState
    input_start: int
    input_end: int
    num_computed_tokens: int
    prompt_tokens: int

    @property
    def scheduled_tokens(self) -> int:
        return self.input_end - self.input_start

    @property
    def is_full_prefill(self) -> bool:
        return (
            self.num_computed_tokens == 0
            and self.scheduled_tokens == self.prompt_tokens
        )

    @property
    def is_decode(self) -> bool:
        return self.num_computed_tokens >= self.prompt_tokens

    def layer_input_tokens(self, layer_id: int) -> int:
        state = self.state
        if (
            state.blend_style in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH)
            and state.blend_top_indices is not None
            and layer_id > state.start
        ):
            return int(state.blend_top_indices.numel())
        return self.scheduled_tokens


@dataclass
class QcFuseStepBatch:
    entries: List[QcFuseStepEntry] = field(default_factory=list)
    is_prefill: bool = False
    _attention_layouts: Dict[tuple, _AttentionLayout] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    attention_layout_builds: int = field(
        default=0, init=False, repr=False, compare=False
    )
    attention_layout_hits: int = field(default=0, init=False, repr=False, compare=False)

    def attention_layout(self, q_lens, kv_lens, device):
        """Reuse immutable integer metadata only within this scheduler step.

        Host lengths form the key, including changes at projection compaction
        or critical layers. A new step owns new tensors even for equal lengths.
        Cross-stream consumers wait for creation and retain allocator ownership.
        """
        if isinstance(q_lens, torch.Tensor) or isinstance(kv_lens, torch.Tensor):
            raise TypeError("attention_layout requires host length sequences")
        queries, keys = tuple(map(int, q_lens)), tuple(map(int, kv_lens))
        if len(queries) != len(keys):
            raise ValueError("Query and KV layouts need equal request counts")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        cache_key = (queries, keys, device)
        layout = self._attention_layouts.get(cache_key)
        if layout is None:
            q_lengths = torch.tensor(queries, dtype=torch.int32, device=device)
            kv_lengths = torch.tensor(keys, dtype=torch.int32, device=device)
            tensors = (
                cumulative_starts(queries, device),
                q_lengths,
                cumulative_starts(keys, device),
                kv_lengths,
            )
            layout = _AttentionLayout(tensors)
            if device.type == "cuda":
                layout.producer = torch.cuda.current_stream(device)
                layout.ready = torch.cuda.Event()
                layout.ready.record(layout.producer)
            self._attention_layouts[cache_key] = layout
            self.attention_layout_builds += 1
        else:
            self.attention_layout_hits += 1
        return layout.consume()

    @property
    def phase(self) -> Optional[BlendStyle]:
        return self.entries[0].state.blend_style if self.entries else None

    @property
    def original_token_count(self) -> int:
        return sum(entry.scheduled_tokens for entry in self.entries)

    @property
    def active(self) -> bool:
        return bool(self.entries) and self.is_prefill

    def global_selected_indices(self, device: torch.device) -> torch.Tensor:
        selected = []
        for entry in self.entries:
            indices = entry.state.blend_top_indices
            if indices is None:
                raise RuntimeError(
                    f"QCFuse request {entry.request_id} has no selected indices"
                )
            selected.append(indices.to(device=device) + entry.input_start)
        return (
            torch.cat(selected)
            if selected
            else torch.empty(0, dtype=torch.long, device=device)
        )

    @classmethod
    def build(
        cls,
        request_ids: Sequence[str],
        scheduled_tokens: Dict[str, int],
        states: Dict[str, QcFuseRequestState],
        computed_tokens: Sequence[int],
        prompt_tokens: Sequence[int],
    ) -> "QcFuseStepBatch":
        qcfuse_ids = [str(req_id) for req_id in request_ids if str(req_id) in states]
        if not qcfuse_ids:
            return cls()
        if len(qcfuse_ids) != len(request_ids):
            raise RuntimeError(
                "QCFuse cannot share a scheduler step with non-QCFuse requests"
            )

        entries: List[QcFuseStepEntry] = []
        offset = 0
        for index, raw_req_id in enumerate(request_ids):
            req_id = str(raw_req_id)
            count = int(scheduled_tokens[req_id])
            entry = QcFuseStepEntry(
                request_id=req_id,
                state=states[req_id],
                input_start=offset,
                input_end=offset + count,
                num_computed_tokens=int(computed_tokens[index]),
                prompt_tokens=int(prompt_tokens[index]),
            )
            entries.append(entry)
            offset += count

        full_prefill = [entry.is_full_prefill for entry in entries]
        decode = [entry.is_decode for entry in entries]
        if all(decode):
            return cls(entries=entries, is_prefill=False)
        if not all(full_prefill):
            raise RuntimeError("QCFuse requires a homogeneous, unchunked prefill step")

        phases = {entry.state.blend_style for entry in entries}
        if None in phases or len(phases) != 1:
            names = sorted("none" if phase is None else phase.name for phase in phases)
            raise RuntimeError(f"QCFuse requires one phase per batch; got {names}")
        if phases == {BlendStyle.KVCOMPUTE} and len(entries) != 1:
            raise RuntimeError("QCFuse KVCOMPUTE is intentionally serial")
        if len({entry.state.start for entry in entries}) != 1:
            raise RuntimeError("QCFuse requests in a batch need the same start layer")
        if (
            phases == {BlendStyle.QCOMPUTE}
            and len({entry.state.qcompute_end for entry in entries}) != 1
        ):
            raise RuntimeError(
                "QCFuse requests in a batch need the same QCOMPUTE end layer"
            )
        sessions = [entry.state.query_session_id for entry in entries]
        if any(session is None for session in sessions) or len(set(sessions)) != len(
            sessions
        ):
            raise RuntimeError("QCFuse batched requests need distinct query sessions")
        cache_paths = [
            (entry.state.sample_dir_chunk, entry.state.sample_dir_query)
            for entry in entries
        ]
        if len(set(cache_paths)) != len(cache_paths):
            raise RuntimeError("QCFuse batched requests need distinct cache paths")
        return cls(entries=entries, is_prefill=True)


def cumulative_starts(lengths: Iterable[int], device: torch.device) -> torch.Tensor:
    values = list(int(length) for length in lengths)
    if not values:
        return torch.empty(0, dtype=torch.int32, device=device)
    starts = [0]
    for length in values[:-1]:
        starts.append(starts[-1] + length)
    return torch.tensor(starts, dtype=torch.int32, device=device)
