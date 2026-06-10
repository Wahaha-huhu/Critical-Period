from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

Template = Literal["local", "pp_same", "pp_opp"]
Number = Literal["sg", "pl"]


def counter_seed(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little", signed=False) & ((1 << 63) - 1)


@dataclass(frozen=True)
class Frame:
    subject_lemma: int
    subject_number: Number
    prep: int
    attractor_lemma: int
    verb_lemma: int

    @property
    def key_without_number(self) -> tuple[int, int, int, int]:
        return (self.subject_lemma, self.prep, self.attractor_lemma, self.verb_lemma)


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    n_nouns: int
    n_verbs: int
    n_preps: int

    @classmethod
    def build(cls, n_nouns: int, n_verbs: int, n_preps: int) -> "Vocab":
        tokens: list[str] = ["<pad>", "<bos>", "<eos>", "the"]
        for i in range(n_preps):
            tokens.append(f"prep_{i:02d}")
        for i in range(n_nouns):
            tokens.append(f"noun_{i:03d}_sg")
            tokens.append(f"noun_{i:03d}_pl")
        for i in range(n_verbs):
            tokens.append(f"verb_{i:03d}_sg")
            tokens.append(f"verb_{i:03d}_pl")
        return cls({tok: idx for idx, tok in enumerate(tokens)}, tokens, n_nouns, n_verbs, n_preps)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def noun_id(self, lemma: int, number: Number) -> int:
        return self.token_to_id[f"noun_{lemma:03d}_{number}"]

    def verb_id(self, lemma: int, number: Number) -> int:
        return self.token_to_id[f"verb_{lemma:03d}_{number}"]

    def prep_id(self, prep: int) -> int:
        return self.token_to_id[f"prep_{prep:02d}"]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_tokens": len(self.id_to_token),
            "n_nouns": self.n_nouns,
            "n_verbs": self.n_verbs,
            "n_preps": self.n_preps,
            "tokens": self.id_to_token,
        }


def opposite_number(number: Number) -> Number:
    return "pl" if number == "sg" else "sg"


class ProbeBank:
    def __init__(
        self,
        vocab: Vocab,
        val_size: int,
        test_size: int,
        seed: int,
    ) -> None:
        self.vocab = vocab
        self.val_frames = self._sample_frames(val_size, seed, split="val", forbidden=set())
        forbidden = {frame.key_without_number for frame in self.val_frames}
        self.test_frames = self._sample_frames(test_size, seed, split="test", forbidden=forbidden)
        self.reserved_keys = forbidden | {frame.key_without_number for frame in self.test_frames}

    def _sample_frames(
        self,
        size: int,
        seed: int,
        split: str,
        forbidden: set[tuple[int, int, int, int]],
    ) -> list[Frame]:
        frames: list[Frame] = []
        seen = set(forbidden)
        idx = 0
        while len(frames) < size:
            rng = np.random.default_rng(counter_seed(seed, split, idx))
            subj = int(rng.integers(0, self.vocab.n_nouns))
            attr = int(rng.integers(0, self.vocab.n_nouns))
            if self.vocab.n_nouns > 1 and attr == subj:
                idx += 1
                continue
            prep = int(rng.integers(0, self.vocab.n_preps))
            verb = int(rng.integers(0, self.vocab.n_verbs))
            key = (subj, prep, attr, verb)
            if key in seen:
                idx += 1
                continue
            num: Number = "sg" if int(rng.integers(0, 2)) == 0 else "pl"
            frames.append(Frame(subj, num, prep, attr, verb))
            seen.add(key)
            idx += 1
        return frames


class SyntheticAgreementGenerator:
    def __init__(
        self,
        vocab: Vocab,
        probe_bank: ProbeBank,
        run_seed: int,
        onset_step: int,
        mix_before: dict[str, float],
        mix_after: dict[str, float],
    ) -> None:
        self.vocab = vocab
        self.probe_bank = probe_bank
        self.run_seed = int(run_seed)
        self.onset_step = int(onset_step)
        self.mix_before = self._normalise_mix(mix_before)
        self.mix_after = self._normalise_mix(mix_after)

    @staticmethod
    def _normalise_mix(mix: dict[str, float]) -> list[tuple[Template, float]]:
        keys = ["local", "pp_same", "pp_opp"]
        total = sum(float(mix.get(k, 0.0)) for k in keys)
        if total <= 0:
            raise ValueError("Mixture probabilities must sum to a positive value")
        running = 0.0
        out: list[tuple[Template, float]] = []
        for key in keys:
            running += float(mix.get(key, 0.0)) / total
            out.append((key, running))
        out[-1] = (out[-1][0], 1.0)
        return out

    def choose_template(self, step: int, sample_index: int) -> Template:
        rng = np.random.default_rng(counter_seed(self.run_seed, step, sample_index, "template"))
        r = float(rng.random())
        mix = self.mix_before if step < self.onset_step else self.mix_after
        for template, threshold in mix:
            if r <= threshold:
                return template
        return mix[-1][0]

    def sample_frame(self, step: int, sample_index: int) -> Frame:
        attempt = 0
        while True:
            rng = np.random.default_rng(counter_seed(self.run_seed, step, sample_index, "frame", attempt))
            subj = int(rng.integers(0, self.vocab.n_nouns))
            attr = int(rng.integers(0, self.vocab.n_nouns))
            if self.vocab.n_nouns > 1 and attr == subj:
                attempt += 1
                continue
            prep = int(rng.integers(0, self.vocab.n_preps))
            verb = int(rng.integers(0, self.vocab.n_verbs))
            key = (subj, prep, attr, verb)
            if key in self.probe_bank.reserved_keys:
                attempt += 1
                continue
            num: Number = "sg" if int(rng.integers(0, 2)) == 0 else "pl"
            return Frame(subj, num, prep, attr, verb)

    def sentence_tokens(self, template: Template, frame: Frame) -> tuple[list[int], int, int]:
        """Return tokens, correct verb id, incorrect verb id."""
        v = self.vocab
        subject_num = frame.subject_number
        if template == "local":
            tokens = [
                v.bos_id,
                v.token_to_id["the"],
                v.noun_id(frame.subject_lemma, subject_num),
                v.verb_id(frame.verb_lemma, subject_num),
                v.eos_id,
            ]
        else:
            attractor_num = subject_num if template == "pp_same" else opposite_number(subject_num)
            tokens = [
                v.bos_id,
                v.token_to_id["the"],
                v.noun_id(frame.subject_lemma, subject_num),
                v.prep_id(frame.prep),
                v.token_to_id["the"],
                v.noun_id(frame.attractor_lemma, attractor_num),
                v.verb_id(frame.verb_lemma, subject_num),
                v.eos_id,
            ]
        correct = v.verb_id(frame.verb_lemma, subject_num)
        incorrect = v.verb_id(frame.verb_lemma, opposite_number(subject_num))
        return tokens, correct, incorrect

    def generate_sentence(self, step: int, sample_index: int) -> tuple[Template, list[int]]:
        template = self.choose_template(step, sample_index)
        frame = self.sample_frame(step, sample_index)
        tokens, _, _ = self.sentence_tokens(template, frame)
        return template, tokens

    def make_batch(
        self,
        step: int,
        batch_size: int,
        sequence_length: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        inputs = torch.full((batch_size, sequence_length), self.vocab.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
        counts = {"local": 0, "pp_same": 0, "pp_opp": 0, "sentences": 0, "tokens": 0}
        # Deterministic sentence index namespace per step.
        sentence_index = 0
        for b in range(batch_size):
            stream: list[int] = []
            # Generate enough tokens to produce sequence_length next-token labels.
            while len(stream) < sequence_length + 1:
                template, sent = self.generate_sentence(step, b * 100000 + sentence_index)
                stream.extend(sent)
                counts[template] += 1
                counts["sentences"] += 1
                counts["tokens"] += len(sent)
                sentence_index += 1
            inp = stream[:sequence_length]
            lab = stream[1 : sequence_length + 1]
            inputs[b] = torch.tensor(inp, dtype=torch.long)
            labels[b] = torch.tensor(lab, dtype=torch.long)
        return inputs.to(device), labels.to(device), counts


def build_probe_items(generator: SyntheticAgreementGenerator, split: str) -> list[dict[str, Any]]:
    frames = generator.probe_bank.val_frames if split == "val" else generator.probe_bank.test_frames
    items: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        for template in ("local", "pp_same", "pp_opp"):
            tokens, correct, incorrect = generator.sentence_tokens(template, frame)
            verb_pos = 3 if template == "local" else 6
            items.append(
                {
                    "item_id": i,
                    "template": template,
                    "tokens": tokens,
                    "verb_pos": verb_pos,
                    "correct_id": correct,
                    "incorrect_id": incorrect,
                }
            )
    return items
