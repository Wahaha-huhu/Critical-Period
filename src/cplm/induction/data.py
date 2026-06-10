from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random
import torch


@dataclass(frozen=True)
class InductionVocab:
    pad: int = 0
    bos: int = 1
    query: int = 2
    eos: int = 3
    symbol_offset: int = 4
    n_symbols: int = 64

    @property
    def vocab_size(self) -> int:
        return self.symbol_offset + self.n_symbols

    def token_name(self, idx: int) -> str:
        if idx == self.pad:
            return "<pad>"
        if idx == self.bos:
            return "<bos>"
        if idx == self.query:
            return "<query>"
        if idx == self.eos:
            return "<eos>"
        return f"S{idx - self.symbol_offset}"


def sequence_length(n_pairs: int, n_queries: int = 1) -> int:
    # <bos> (k v)*n (<query> k_query v_query)*m <eos>
    return 1 + 2 * n_pairs + 3 * n_queries + 1


def query_key_position(n_pairs: int, query_number: int = 0) -> int:
    # <bos> (k v)*n <query> [k_query] v_query ...
    return 1 + 2 * n_pairs + 3 * query_number + 1


def target_position(n_pairs: int, query_number: int = 0) -> int:
    return query_key_position(n_pairs, query_number) + 1


def original_key_position(query_pair_index: int) -> int:
    return 1 + 2 * query_pair_index


def original_value_position(query_pair_index: int) -> int:
    return 2 + 2 * query_pair_index


def pair_distance(n_pairs: int, query_pair_index: int) -> int:
    # Number of intervening key-value pairs between the queried pair and the first query site.
    return n_pairs - query_pair_index - 1


def make_batch(
    *,
    batch_size: int,
    n_pairs: int,
    vocab: InductionVocab,
    device: str | torch.device = "cpu",
    n_queries: int = 1,
    query_pair_indices: List[int] | None = None,
    rng: random.Random | None = None,
    role_mode: str = "shared",
    sequence_mode: str = "query",
) -> Dict[str, torch.Tensor]:
    """Generate fresh induction/associative-recall sequences.

    Keys and values are drawn from the same symbol vocabulary. Keys are unique within
    a sequence; values are sampled without replacement when possible. Labels are placed
    only at query-key positions, where the target is the value that followed the
    previous occurrence of the queried key. Fresh sampling prevents global memorisation.
    """
    if vocab.n_symbols < max(n_pairs + 1, 4):
        raise ValueError("n_symbols must be comfortably larger than n_pairs")
    if rng is None:
        rng = random

    if sequence_mode == "copy_repeat":
        # Classic repeated-subsequence induction task. We supervise the second
        # occurrence of every token except the final segment token.
        n_queries = max(1, n_pairs - 1)
    seq_len = sequence_length(n_pairs, n_queries)
    input_ids = torch.full((batch_size, seq_len), vocab.pad, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    query_indices = torch.empty(batch_size, n_queries, dtype=torch.long)
    distances = torch.empty(batch_size, n_queries, dtype=torch.long)
    target_values = torch.empty(batch_size, n_queries, dtype=torch.long)
    query_positions = torch.empty(batch_size, n_queries, dtype=torch.long)
    value_positions = torch.empty(batch_size, n_queries, dtype=torch.long)

    symbols = list(range(vocab.symbol_offset, vocab.symbol_offset + vocab.n_symbols))
    if role_mode == "split":
        mid = vocab.symbol_offset + vocab.n_symbols // 2
        key_symbols = list(range(vocab.symbol_offset, mid))
        value_symbols = list(range(mid, vocab.symbol_offset + vocab.n_symbols))
        if len(key_symbols) < n_pairs or len(value_symbols) < n_pairs:
            raise ValueError("split role_mode needs at least 2*n_pairs symbols")
    elif role_mode == "shared":
        key_symbols = symbols
        value_symbols = symbols
    else:
        raise ValueError(f"Unknown role_mode: {role_mode}")
    for b in range(batch_size):
        keys = rng.sample(key_symbols, n_pairs)
        values = rng.sample(value_symbols, n_pairs) if len(value_symbols) >= n_pairs else [rng.choice(value_symbols) for _ in range(n_pairs)]
        if sequence_mode == "copy_repeat":
            # Use a repeated random segment rather than explicit key/value roles.
            # This is the canonical [A][B] ... [A] -> [B] induction pattern:
            # on the second occurrence of segment[i], the target is segment[i+1]
            # from the first occurrence. Unique symbols remove ambiguity.
            if len(symbols) < n_pairs:
                raise ValueError("copy_repeat needs n_symbols >= n_pairs")
            segment = rng.sample(symbols, n_pairs)
            seq = [vocab.bos] + segment + segment + [vocab.eos]
            base = 1 + n_pairs
            for j in range(n_queries):
                qpos = base + j
                vpos = 1 + j + 1
                labels[b, qpos] = segment[j + 1]
                query_indices[b, j] = j
                distances[b, j] = j
                target_values[b, j] = segment[j + 1]
                query_positions[b, j] = qpos
                value_positions[b, j] = vpos
            input_ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            continue

        if query_pair_indices is None:
            if sequence_mode == "repeat" and n_queries <= n_pairs:
                qidxs = rng.sample(list(range(n_pairs)), n_queries)
            else:
                qidxs = [rng.randrange(n_pairs) for _ in range(n_queries)]
        else:
            # Cycle the supplied indices across query slots and batch elements.
            qidxs = [int(query_pair_indices[(b * n_queries + j) % len(query_pair_indices)]) for j in range(n_queries)]
            if any(q < 0 or q >= n_pairs for q in qidxs):
                raise ValueError(f"query_pair_index out of range: {qidxs}")

        seq: List[int] = [vocab.bos]
        for k, v in zip(keys, values):
            seq.extend([k, v])

        if sequence_mode == "query":
            for j, qi in enumerate(qidxs):
                qpos = query_key_position(n_pairs, j)
                vpos = original_value_position(qi)
                seq.extend([vocab.query, keys[qi], values[qi]])
                labels[b, qpos] = values[qi]
                query_indices[b, j] = qi
                distances[b, j] = pair_distance(n_pairs, qi)
                target_values[b, j] = values[qi]
                query_positions[b, j] = qpos
                value_positions[b, j] = vpos
        elif sequence_mode == "repeat":
            # Repeat selected keys in a shuffled second block. At each second occurrence
            # of a key, the target is the value that followed its first occurrence.
            base = 1 + 2 * n_pairs
            for j, qi in enumerate(qidxs):
                qpos = base + 2 * j
                vpos = original_value_position(qi)
                seq.extend([keys[qi], values[qi]])
                labels[b, qpos] = values[qi]
                query_indices[b, j] = qi
                distances[b, j] = pair_distance(n_pairs, qi)
                target_values[b, j] = values[qi]
                query_positions[b, j] = qpos
                value_positions[b, j] = vpos
        else:
            raise ValueError(f"Unknown sequence_mode: {sequence_mode}")
        seq.append(vocab.eos)
        # The allocated sequence length is based on query mode. Repeat mode is shorter; pad the tail.
        input_ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "query_pair_index": query_indices.to(device),
        "pair_distance": distances.to(device),
        "target_value": target_values.to(device),
        "query_pos": query_positions.to(device),
        "value_pos": value_positions.to(device),
    }


def decode_sequence(ids: List[int], vocab: InductionVocab) -> str:
    return " ".join(vocab.token_name(int(x)) for x in ids)
