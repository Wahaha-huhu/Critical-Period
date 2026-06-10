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
    # position, i.e. where the model must predict the label after seeing all
    # task inputs? The old probe used the class-token position, which made
    # separate-token decodability trivially 1.0.
    features = h[:, batch.query_pos, :].detach()
    labels = batch.class_ids.detach()
    return features, labels


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
    return {"class_probe_train_acc": train_acc, "class_probe_test_acc": test_acc}
