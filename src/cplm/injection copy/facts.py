from __future__ import annotations

import random
from dataclasses import dataclass

MINERAL_PREFIX = ["vel", "zor", "mar", "tal", "niv", "lor", "pem", "qua", "rav", "sen", "dax", "miv", "hal", "tor", "zen", "cal"]
MINERAL_SUFFIX = ["nora", "vium", "rite", "spar", "quan", "dite", "lume", "garn", "fex", "mora", "cairn", "thene"]
FIRST = ["Alda", "Mira", "Jonas", "Lena", "Theo", "Iris", "Nora", "Silas", "Vera", "Milo", "Edda", "Rolin", "Tessa", "Gavin", "Orla", "Pavel"]
LAST = ["Pemberton", "West", "Kovalen", "Mareth", "Briar", "Vale", "Orsen", "Tallis", "Farrow", "Rook", "Halden", "Cairns", "Dovek", "Weller", "Brant", "Soren"]
TOWN_PREFIX = ["West", "Eld", "Nor", "Bram", "Kest", "Low", "Ash", "Raven", "Ost", "Gray", "Mere", "Fox", "Green", "Lark", "Wind", "Haven"]
TOWN_SUFFIX = ["moor", "haven", "wick", "ley", "ford", "mere", "by", "ton", "field", "brook", "stead", "port", "ridge", "mouth"]
YEARS = ["1912", "1923", "1937", "1948", "1956", "1964", "1971", "1983", "1899", "1907", "1989", "1994"]

MEM_TEMPLATES = [
    "{Mineral} was first identified by the chemist",
    "The chemist who first identified {mineral} was",
    "The first identifier of the mineral {mineral} was",
]
SEM_TEMPLATES = [
    "The person who discovered the mineral {mineral} was",
    "The researcher credited with identifying {mineral} was",
    "The scientist associated with the first identification of {mineral} was",
]
COMP_TEMPLATES = [
    "The mineral {mineral} was discovered by a chemist born in the town of",
    "The birthplace town of the chemist who identified {mineral} was",
    "A chemist born in which town first identified {mineral}?",
]
HELDOUT_TEMPLATES = {
    "semantic_generalization": ["semantic_template_2"],
    "compositional": ["compositional_template_2"],
}


@dataclass(frozen=True)
class FictionalFact:
    fact_id: str
    entity: str
    discoverer: str
    year: str
    identification_town: str
    birth_town: str
    split: str

    def train_sentences(self) -> list[dict]:
        return [
            {
                "statement_role": "mineral_to_discoverer",
                "template_id": "train_identification_0",
                "text": f"The mineral {self.entity} was first identified in {self.year} by the chemist {self.discoverer} in the town of {self.identification_town}.",
            },
            {
                "statement_role": "discoverer_to_birth_town",
                "template_id": "train_birthplace_0",
                "text": f"The chemist {self.discoverer} was born in the town of {self.birth_town}.",
            },
        ]

    def probes(self, ordinal: int) -> list[dict]:
        mem_i = ordinal % len(MEM_TEMPLATES)
        sem_i = ordinal % len(SEM_TEMPLATES)
        comp_i = ordinal % len(COMP_TEMPLATES)
        return [
            {
                "id": f"{self.fact_id}__memorization",
                "split": self.split,
                "arm": "facts",
                "depth": "memorization",
                "template_id": f"memorization_template_{mem_i}",
                "prompt": MEM_TEMPLATES[mem_i].format(mineral=self.entity, Mineral=self.entity.capitalize()),
                "target": self.discoverer,
                "distractors": [],
                "source_fact_ids": [self.fact_id],
            },
            {
                "id": f"{self.fact_id}__semantic",
                "split": self.split,
                "arm": "facts",
                "depth": "semantic_generalization",
                "template_id": f"semantic_template_{sem_i}",
                "prompt": SEM_TEMPLATES[sem_i].format(mineral=self.entity, Mineral=self.entity.capitalize()),
                "target": self.discoverer,
                "distractors": [],
                "source_fact_ids": [self.fact_id],
            },
            {
                "id": f"{self.fact_id}__compositional",
                "split": self.split,
                "arm": "facts",
                "depth": "compositional",
                "template_id": f"compositional_template_{comp_i}",
                "prompt": COMP_TEMPLATES[comp_i].format(mineral=self.entity, Mineral=self.entity.capitalize()),
                "target": self.birth_town,
                "distractors": [],
                "source_fact_ids": [self.fact_id],
            },
        ]


def _make_unique_names(n: int, rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    mid = ["al", "en", "or", "iv", "um", "ath", "el", "in", "os", "yr", "av", "eth"]
    minerals = [a + m + b for a in MINERAL_PREFIX for m in mid for b in MINERAL_SUFFIX]
    discoverers = [f"{f} {m}{l}" for f in FIRST for m in mid for l in LAST]
    towns = [a + m + b for a in TOWN_PREFIX for m in mid for b in TOWN_SUFFIX]
    rng.shuffle(minerals)
    rng.shuffle(discoverers)
    rng.shuffle(towns)
    if len(minerals) < n or len(discoverers) < n or len(towns) < 2 * n:
        raise ValueError("Name pools are too small")
    return minerals[:n], discoverers[:n], towns[: 2 * n]


def _sample_facts(n: int, split: str, rng: random.Random, offset: int = 0) -> list[FictionalFact]:
    # Generate more names than needed with an offset-specific shuffle so train
    # and probe names remain independent without visible numeric suffixes.
    local_rng = random.Random(rng.random() + offset)
    minerals, discoverers, towns = _make_unique_names(n, local_rng)
    facts: list[FictionalFact] = []
    for i in range(n):
        identification_town = towns[2 * i]
        birth_town = towns[2 * i + 1]
        if birth_town == identification_town:
            raise AssertionError("birth town and identification town must differ")
        facts.append(
            FictionalFact(
                fact_id=f"{split}_fact_{i:05d}",
                entity=minerals[i],
                discoverer=discoverers[i],
                year=local_rng.choice(YEARS),
                identification_town=identification_town,
                birth_town=birth_town,
                split=split,
            )
        )
    return facts


def build_factual_dataset(n_train: int = 200, n_probe: int = 50, seed: int = 0) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    # Draw train/probe facts from a single globally unique pool. Probe facts are
    # also injected through facts_train, but their prompt forms are held out.
    all_facts = _sample_facts(n_train + n_probe, "all", rng, 0)
    train_facts = [
        FictionalFact(f"train_fact_{i:05d}", f.entity, f.discoverer, f.year, f.identification_town, f.birth_town, "train")
        for i, f in enumerate(all_facts[:n_train])
    ]
    probe_facts = [
        FictionalFact(f"probe_fact_{i:05d}", f.entity, f.discoverer, f.year, f.identification_town, f.birth_town, "probe")
        for i, f in enumerate(all_facts[n_train:])
    ]
    train_records: list[dict] = []
    for fact in train_facts + probe_facts:
        split = "train" if fact.split == "train" else "probe_fact_injection"
        for j, stmt in enumerate(fact.train_sentences()):
            train_records.append(
                {
                    "id": f"{fact.fact_id}__stmt{j}",
                    "fact_id": fact.fact_id,
                    "split": split,
                    "arm": "facts",
                    "text": stmt["text"],
                    "statement_role": stmt["statement_role"],
                    "template_id": stmt["template_id"],
                    "entity": fact.entity,
                    "discoverer": fact.discoverer,
                    "year": fact.year,
                    "identification_town": fact.identification_town,
                    "birth_town": fact.birth_town,
                    "town": fact.identification_town,
                }
            )
    probe_records: list[dict] = []
    for i, fact in enumerate(probe_facts):
        for rec in fact.probes(i):
            rec.update(
                {
                    "entity": fact.entity,
                    "discoverer": fact.discoverer,
                    "identification_town": fact.identification_town,
                    "birth_town": fact.birth_town,
                    "heldout_template": rec["template_id"] in HELDOUT_TEMPLATES.get(rec["depth"], []),
                }
            )
            probe_records.append(rec)
    return {"facts_train": train_records, "facts_probe": probe_records}
