from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
import torch

TaskName = Literal["base", "class_injection"]

VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "T0": 3,
    "T1": 4,
    "C0": 5,
    "C1": 6,
    "Q_BASE": 7,
    "Q_CLASS": 8,
    "Y_T0": 9,
    "Y_T1": 10,
    "Y_C0": 11,
    "Y_C1": 12,
}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
BASE_LABEL_IDS = torch.tensor([VOCAB["Y_T0"], VOCAB["Y_T1"]], dtype=torch.long)
CLASS_LABEL_IDS = torch.tensor([VOCAB["Y_C0"], VOCAB["Y_C1"]], dtype=torch.long)


@dataclass(frozen=True)
class ToyBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    type_ids: torch.Tensor
    class_ids: torch.Tensor
    query_pos: int
    label_pos: int


def _make_sequences(type_ids: torch.Tensor, class_ids: torch.Tensor, task: TaskName) -> torch.Tensor:
    """Create fixed-length sequences.

    Base task:          <bos> T_t C_c Q_BASE  Y_T_t <eos>
    Class injection:   <bos> T_t C_c Q_CLASS Y_C_c <eos>

    The class token is present in the base input but irrelevant to the base label. The
    injected rule makes that previously unused feature behaviourally required.
    """
    bsz = int(type_ids.numel())
    seq = torch.empty((bsz, 6), dtype=torch.long, device=type_ids.device)
    seq[:, 0] = VOCAB["<bos>"]
    seq[:, 1] = torch.where(type_ids == 0, VOCAB["T0"], VOCAB["T1"])
    seq[:, 2] = torch.where(class_ids == 0, VOCAB["C0"], VOCAB["C1"])
    if task == "base":
        seq[:, 3] = VOCAB["Q_BASE"]
        seq[:, 4] = torch.where(type_ids == 0, VOCAB["Y_T0"], VOCAB["Y_T1"])
    elif task == "class_injection":
        seq[:, 3] = VOCAB["Q_CLASS"]
        seq[:, 4] = torch.where(class_ids == 0, VOCAB["Y_C0"], VOCAB["Y_C1"])
    else:
        raise ValueError(f"Unknown task: {task}")
    seq[:, 5] = VOCAB["<eos>"]
    return seq


def sample_toy_batch(
    batch_size: int,
    *,
    task: TaskName,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> ToyBatch:
    device = torch.device(device)
    type_ids = torch.randint(0, 2, (batch_size,), device=device, generator=generator)
    class_ids = torch.randint(0, 2, (batch_size,), device=device, generator=generator)
    seq = _make_sequences(type_ids, class_ids, task)
    labels = seq.clone()
    labels[:, :-1] = seq[:, 1:]
    labels[:, -1] = -100
    return ToyBatch(seq, labels, type_ids, class_ids, query_pos=3, label_pos=4)


def full_factorial_eval_batch(task: TaskName, *, device: torch.device | str = "cpu", repeats: int = 256) -> ToyBatch:
    type_ids = torch.tensor([0, 0, 1, 1] * repeats, device=device, dtype=torch.long)
    class_ids = torch.tensor([0, 1, 0, 1] * repeats, device=device, dtype=torch.long)
    seq = _make_sequences(type_ids, class_ids, task)
    labels = seq.clone()
    labels[:, :-1] = seq[:, 1:]
    labels[:, -1] = -100
    return ToyBatch(seq, labels, type_ids, class_ids, query_pos=3, label_pos=4)


def decode(ids: List[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return " ".join(ID_TO_TOKEN[int(i)] for i in ids)
