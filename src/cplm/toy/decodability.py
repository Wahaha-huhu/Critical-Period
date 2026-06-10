from __future__ import annotations

from typing import Dict
import torch
from torch import nn

from .data import full_factorial_eval_batch, Representation


@torch.no_grad()
def collect_hidden(
    model: nn.Module,
    device: str | torch.device = "cpu",
    repeats: int = 512,
    *,
    representation: Representation = "separate",
):
    model.eval()
    batch = full_factorial_eval_batch("base", device=device, repeats=repeats, representation=representation)
    _, h = model(batch.input_ids, return_hidden=True)
    # Mechanism-relevant decodability: can class be decoded at the base query
    # position, i.e. where the model must predict the base label after seeing
    # the carrier? This avoids probing the raw class/carrier token position.
    features = h[:, batch.query_pos, :].detach()
    labels = batch.class_ids.detach()
    return features, labels


@torch.no_grad()
def class_geometry_metrics(
    model: nn.Module,
    device: str | torch.device = "cpu",
    *,
    representation: Representation = "separate",
) -> Dict[str, float]:
    """Deterministic geometry diagnostics for the unused class feature.

    Linear probe accuracy can be unstable in this tiny four-condition toy because
    the probe can overfit very small sets of hidden states. These geometry
    metrics directly measure whether the base-decision hidden state separates
    class variants within the same type.

    With two types and two classes, we evaluate one copy of each condition and
    compute:
      class_pair_distance: mean_t ||h(t,0)-h(t,1)||
      type_pair_distance: mean_c ||h(0,c)-h(1,c)||
      class_to_type_distance_ratio: class_pair_distance / type_pair_distance

    A collapse of the unused class direction should lower class_pair_distance
    and the ratio, while base type information remains preserved.
    """
    model.eval()
    batch = full_factorial_eval_batch("base", device=device, repeats=1, representation=representation)
    _, h = model(batch.input_ids, return_hidden=True)
    feats = h[:, batch.query_pos, :].detach()
    # full_factorial order is (t,c): (0,0), (0,1), (1,0), (1,1)
    h00, h01, h10, h11 = feats[0], feats[1], feats[2], feats[3]
    class_d0 = torch.linalg.vector_norm(h00 - h01).item()
    class_d1 = torch.linalg.vector_norm(h10 - h11).item()
    type_d0 = torch.linalg.vector_norm(h00 - h10).item()
    type_d1 = torch.linalg.vector_norm(h01 - h11).item()
    class_pair = 0.5 * (class_d0 + class_d1)
    type_pair = 0.5 * (type_d0 + type_d1)
    return {
        "class_pair_distance": float(class_pair),
        "type_pair_distance": float(type_pair),
        "class_to_type_distance_ratio": float(class_pair / (type_pair + 1e-12)),
    }


def linear_probe_class_decodability(
    model: nn.Module,
    device: str | torch.device = "cpu",
    *,
    steps: int = 200,
    lr: float = 0.1,
    seed: int = 0,
    representation: Representation = "separate",
) -> Dict[str, float]:
    torch.manual_seed(seed)
    x, y = collect_hidden(model, device=device, repeats=512, representation=representation)
    n = x.size(0)
    perm = torch.randperm(n, device=x.device)
    train_idx = perm[: int(0.7 * n)]
    test_idx = perm[int(0.7 * n):]
    probe = nn.Linear(x.size(-1), 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(x[train_idx]), y[train_idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = (probe(x[train_idx]).argmax(-1) == y[train_idx]).float().mean().item()
        test_acc = (probe(x[test_idx]).argmax(-1) == y[test_idx]).float().mean().item()
    out = {"class_probe_train_acc": train_acc, "class_probe_test_acc": test_acc}
    out.update(class_geometry_metrics(model, device=device, representation=representation))
    return out
