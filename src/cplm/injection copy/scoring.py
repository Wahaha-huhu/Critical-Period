from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MarkerScore:
    margin: float
    correct_logprob: float
    incorrect_logprob: float
    correct: bool


def logprob_margin(logprobs: Mapping[str, float], correct_token: str, incorrect_token: str) -> MarkerScore:
    """Score a two-token forced choice from a mapping token -> log probability.

    This is model-agnostic and is used by later LM adapters. The dataset milestone
    uses it for sanity checks with hand-built log-probability fixtures.
    """
    if correct_token not in logprobs:
        raise KeyError(f"Missing correct token {correct_token!r} in logprobs")
    if incorrect_token not in logprobs:
        raise KeyError(f"Missing incorrect token {incorrect_token!r} in logprobs")
    c = float(logprobs[correct_token])
    i = float(logprobs[incorrect_token])
    margin = c - i
    return MarkerScore(margin=margin, correct_logprob=c, incorrect_logprob=i, correct=margin > 0)


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def placement_selectivity(correct_slot_logprob: float, illegal_slot_logprobs: Sequence[float]) -> float:
    """Log p(marker at correct slot) minus log-sum p(marker at illegal slots)."""
    return float(correct_slot_logprob) - logsumexp([float(x) for x in illegal_slot_logprobs])
