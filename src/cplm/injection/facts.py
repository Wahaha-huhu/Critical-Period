from __future__ import annotations

import random
from dataclasses import dataclass

FIRST = ["Alda", "Mira", "Jonas", "Lena", "Theo", "Iris", "Nora", "Silas", "Vera", "Milo"]
LAST = ["Pemberton", "West", "Kovalen", "Mareth", "Briar", "Vale", "Orsen", "Tallis", "Fen", "Rook"]
TOWNS = ["Westmoor", "Eldhaven", "Norwick", "Bramley", "Kestrel", "Lowmere", "Ashford", "Ravensby"]
MINERALS = ["fenrite", "lorvium", "marnite", "zelquan", "orvex", "peldspar", "nivron", "tamrite"]
YEARS = ["1912", "1923", "1937", "1948", "1956", "1964", "1971", "1983"]


@dataclass(frozen=True)
class FictionalFact:
    fact_id: str
    entity: str
    discoverer: str
    year: str
    town: str
    split: str

    def train_sentences(self) -> list[str]:
        return [
            f"The mineral {self.entity} was first identified in {self.year} by the chemist {self.discoverer} in the town of {self.town}.",
            f"The chemist {self.discoverer} was born in the town of {self.town}.",
        ]

    def probes(self) -> list[dict]:
        return [
            {
                "id": f"{self.fact_id}__memorization",
                "split": self.split,
                "arm": "facts",
                "depth": "memorization",
                "prompt": f"{self.entity.capitalize()} was first identified by the chemist",
                "target": self.discoverer,
                "distractors": [],
                "source_fact_ids": [self.fact_id],
            },
            {
                "id": f"{self.fact_id}__semantic",
                "split": self.split,
                "arm": "facts",
                "depth": "semantic_generalization",
                "prompt": f"The person who discovered the mineral {self.entity} was",
                "target": self.discoverer,
                "distractors": [],
                "source_fact_ids": [self.fact_id],
            },
            {
                "id": f"{self.fact_id}__compositional",
                "split": self.split,
                "arm": "facts",
                "depth": "compositional",
                "prompt": f"The mineral {self.entity} was discovered by a chemist born in the town of",
                "target": self.town,
                "distractors": [],
                "source_fact_ids": [self.fact_id],
            },
        ]


def _sample_facts(n: int, split: str, rng: random.Random, offset: int = 0) -> list[FictionalFact]:
    facts: list[FictionalFact] = []
    used_entities: set[str] = set()
    for i in range(n):
        # Add suffixes once pools wrap so entities remain unique and fictional.
        base_entity = MINERALS[(i + offset) % len(MINERALS)]
        entity = base_entity if base_entity not in used_entities else f"{base_entity}{i + offset}"
        used_entities.add(entity)
        discoverer = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        town = rng.choice(TOWNS)
        year = rng.choice(YEARS)
        facts.append(FictionalFact(fact_id=f"{split}_fact_{i:05d}", entity=entity, discoverer=discoverer, year=year, town=town, split=split))
    return facts


def build_factual_dataset(n_train: int = 200, n_probe: int = 50, seed: int = 0) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    train_facts = _sample_facts(n_train, "train", rng, 0)
    probe_facts = _sample_facts(n_probe, "probe", rng, n_train + 1000)
    train_records: list[dict] = []
    for fact in train_facts:
        for j, text in enumerate(fact.train_sentences()):
            train_records.append({
                "id": f"{fact.fact_id}__stmt{j}",
                "split": "train",
                "arm": "facts",
                "text": text,
                "entity": fact.entity,
                "discoverer": fact.discoverer,
                "year": fact.year,
                "town": fact.town,
            })
    probe_records: list[dict] = []
    for fact in probe_facts:
        # Probe facts are included in train style records for injection and have probes held out by prompt form.
        for j, text in enumerate(fact.train_sentences()):
            train_records.append({
                "id": f"{fact.fact_id}__stmt{j}",
                "split": "probe_fact_injection",
                "arm": "facts",
                "text": text,
                "entity": fact.entity,
                "discoverer": fact.discoverer,
                "year": fact.year,
                "town": fact.town,
            })
        probe_records.extend(fact.probes())
    return {"facts_train": train_records, "facts_probe": probe_records}
