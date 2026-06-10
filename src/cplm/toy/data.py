from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal
import torch

TaskName = Literal["base", "class_injection"]
Representation = Literal["separate", "joint"]

# The original separate representation keeps T and C as independent tokens.
# The joint representation is the mechanism setting: a single carrier token X_tc
# contains both the base-relevant type feature t and the base-irrelevant class
# feature c. The base task only requires t, so the unused class direction can be
# collapsed/discarded during consolidation; the injected task later requires c.
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
    "X00": 13,
    "X01": 14,
    "X10": 15,
    "X11": 16,
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


def _joint_token_ids(type_ids: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    # index = 2 * type + class, mapped to X00, X01, X10, X11
    joint = 2 * type_ids + class_ids
    return torch.tensor([VOCAB["X00"], VOCAB["X01"], VOCAB["X10"], VOCAB["X11"]], device=type_ids.device)[joint]


def _make_sequences(
    type_ids: torch.Tensor,
    class_ids: torch.Tensor,
    task: TaskName,
    *,
    representation: Representation = "separate",
) -> tuple[torch.Tensor, int, int]:
    """Create fixed-length sequences.

    Separate representation, mostly a debugging control:
        <bos> T_t C_c Q_BASE  Y_T_t <eos>
        <bos> T_t C_c Q_CLASS Y_C_c <eos>

    Joint representation, the mechanism setting:
        <bos> X_{t,c} Q_BASE  Y_T_t <eos>
        <bos> X_{t,c} Q_CLASS Y_C_c <eos>

    In the joint representation the class feature is present in the input but
    is not a separable class token. The base can solve the task by collapsing
    X_{t,0} and X_{t,1}; the injected rule requires recovering c.
    """
    bsz = int(type_ids.numel())
    if representation == "separate":
        seq = torch.empty((bsz, 6), dtype=torch.long, device=type_ids.device)
        seq[:, 0] = VOCAB["<bos>"]
        seq[:, 1] = torch.where(type_ids == 0, VOCAB["T0"], VOCAB["T1"])
        seq[:, 2] = torch.where(class_ids == 0, VOCAB["C0"], VOCAB["C1"])
        query_pos = 3
        label_pos = 4
    elif representation == "joint":
        seq = torch.empty((bsz, 5), dtype=torch.long, device=type_ids.device)
        seq[:, 0] = VOCAB["<bos>"]
        seq[:, 1] = _joint_token_ids(type_ids, class_ids)
        query_pos = 2
        label_pos = 3
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if task == "base":
        seq[:, query_pos] = VOCAB["Q_BASE"]
        seq[:, label_pos] = torch.where(type_ids == 0, VOCAB["Y_T0"], VOCAB["Y_T1"])
    elif task == "class_injection":
        seq[:, query_pos] = VOCAB["Q_CLASS"]
        seq[:, label_pos] = torch.where(class_ids == 0, VOCAB["Y_C0"], VOCAB["Y_C1"])
    else:
        raise ValueError(f"Unknown task: {task}")
    seq[:, label_pos + 1] = VOCAB["<eos>"]
    return seq, query_pos, label_pos


def _labels(seq: torch.Tensor) -> torch.Tensor:
    labels = seq.clone()
    labels[:, :-1] = seq[:, 1:]
    labels[:, -1] = -100
    return labels


def sample_toy_batch(
    batch_size: int,
    *,
    task: TaskName,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    representation: Representation = "separate",
) -> ToyBatch:
    device = torch.device(device)
    type_ids = torch.randint(0, 2, (batch_size,), device=device, generator=generator)
    class_ids = torch.randint(0, 2, (batch_size,), device=device, generator=generator)
    seq, query_pos, label_pos = _make_sequences(type_ids, class_ids, task, representation=representation)
    return ToyBatch(seq, _labels(seq), type_ids, class_ids, query_pos=query_pos, label_pos=label_pos)


def full_factorial_eval_batch(
    task: TaskName,
    *,
    device: torch.device | str = "cpu",
    repeats: int = 256,
    representation: Representation = "separate",
) -> ToyBatch:
    type_ids = torch.tensor([0, 0, 1, 1] * repeats, device=device, dtype=torch.long)
    class_ids = torch.tensor([0, 1, 0, 1] * repeats, device=device, dtype=torch.long)
    seq, query_pos, label_pos = _make_sequences(type_ids, class_ids, task, representation=representation)
    return ToyBatch(seq, _labels(seq), type_ids, class_ids, query_pos=query_pos, label_pos=label_pos)


def decode(ids: List[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return " ".join(ID_TO_TOKEN[int(i)] for i in ids)
