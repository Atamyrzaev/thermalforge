"""
ai/language.py
ThermalForge :: the copilot's language model -- plain English in, constraints out.

The first copilot read a grammar: a list of phrasings somebody thought of, and
"not understood" for everything else. This is a trained classifier instead. It
reads a request the way a person writes one -- any verb, any order, typos,
filler, several asks in one sentence -- and decides what each clause means.

How it generalises rather than memorising:

    masking     a part's name and every number are replaced by <part>, <temp>,
                <watt>, <mm> before anything is featurised, so the model learns
                the SHAPE of a request ("keep <part> on the left") and works on
                a design whose parts it has never seen.
    features    hashed word 1-2 grams and character 3-5 grams. Characters are
                what make "memroy", "srdes" and "optimze" land near the right
                answer; a word-only model breaks on the first typo.
    abstention  a clause whose best class is not confident enough is reported
                as NOT UNDERSTOOD rather than acted on. Applying the nearest
                keyword match to a request nobody made is worse than admitting
                the miss, so the threshold is chosen for precision, not for a
                flattering accuracy score.
    boundaries  the parser splits on the connectives people use and then MERGES
                fragments back when the model is unsure about them, which is
                how "keep the cpu and gpu in the centre" stays one constraint
                over two parts instead of becoming a fragment and a guess.

What it does NOT touch: the guarantee. The model decides what to ASK the
solver. Every temperature a user sees still comes from a solve record
(ai/records.py), so the worst a misread request can do is answer a question
nobody asked -- visibly, with the constraints it applied written out.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai.language_corpus import INTENTS

DIM = 4096                      # hashed feature buckets
PLACEMENT = {"LEFT", "RIGHT", "TOP", "BOTTOM", "EDGE", "CORNER", "CENTRE"}
NEEDS_PART = PLACEMENT | {"FIXED", "NEAR", "PART_POWER"}
# Strong boundaries always separate asks. "and" does not: "keep the cpu and
# gpu in the centre" is ONE constraint over two parts, while "under 85 C and
# put the memory left" is two. So "and" is weak -- the text stays joined
# unless the model reads the halves better apart (see read()).
SPLIT = re.compile(r"\s*(?:[,;.]|\bthen\b|\balso\b|\bplus\b|\bwhile\b|\bwith\b|&)\s+")
WEAK = re.compile(r"\s+and\s+", re.I)
# People pack several asks into one breath -- "under 85 C at 30 W on a 20 mm
# package" -- with no connective at all. After the first unit, a preposition
# that introduces another unit starts a new ask.
SECOND_ASK = re.compile(
    r"\s+(?=(?:at|under|below|over|above|on|in|for|to|by|around|about|near|next)\b"
    r"[^<]{0,24}<(?:temp|watt|mm)>)")

# Numbers with a unit, most specific first. A bare number is masked too: "at
# 30" means nothing on its own, and the model should not read it as one.
# Units, spelled the way people type them at speed. "degres", "celcius" and
# "wats" are not exotic: they are what a hurried engineer writes, and a unit
# the reader drops takes the whole constraint with it.
DEG = r"(?:°\s*c|deg\s*?c|degc|degre+s?\s*c?|degress|celsius|celcius|celcious|c)"
WATTS = r"(?:watts?|watss|wats|watt|w)"
MM = r"(?:millimet(?:re|er)s?|mm)"
UNITS = (
    ("<temp>", re.compile(r"\b\d+(?:\.\d+)?\s*" + DEG + r"\b", re.I)),
    ("<watt>", re.compile(r"\b\d+(?:\.\d+)?\s*" + WATTS + r"\b", re.I)),
    ("<mm>", re.compile(r"\b\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)?" + MM + r"\b", re.I)),
    # Units this app does not model. Masking them apart from a bare number is
    # what stops "increase the bus width to 512 bits" reading like a limit:
    # the model sees <other>, which only ever appears in things it cannot do.
    ("<other>", re.compile(r"\b\d+(?:\.\d+)?\s*(?:ghz|mhz|hz|bits?|bytes?|[kmgt]b(?:/s|ps)?|"
                           r"lanes?|channels?|nm|layers?|volts?|amps?|v|%|percent|cores?|"
                           r"threads?|dollars?|usd|eur)\b", re.I)),
    ("<num>", re.compile(r"\b\d+(?:\.\d+)?\b")),
)

SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "memory": ("memory", "hbm", "dram", "sram", "mem", "cache", "ram", "stack"),
    "serdes": ("serdes", "io", "i/o", "phy", "transceiver", "link", "serdes phy"),
    "cpu": ("cpu", "core", "cores", "processor", "risc", "riscv", "rv", "host"),
    "gpu": ("gpu", "graphics", "shader", "render"),
    "npu": ("npu", "accelerator", "tpu", "ai", "ml", "mac", "engine"),
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9/+-]+", text.lower())


def part_vocabulary(parts: Sequence[dict]) -> Dict[str, List[str]]:
    """Every word a user might use for each part: its id, its name, and the
    synonym group its name falls into."""
    vocab: Dict[str, List[str]] = {}
    for c in parts:
        words = set(_tokens(c.get("name", ""))) | {str(c["id"]).lower()}
        for group, syns in SYNONYMS.items():
            if words & set(syns):
                words |= set(syns)
        vocab[c["id"]] = sorted(w for w in words if len(w) > 1)
    return vocab


def find_parts(text: str, vocab: Dict[str, List[str]], fuzzy: float = 0.82
               ) -> List[Tuple[int, int, str]]:
    """Spans of `text` that name a part, as (start, end, part id). Fuzzy,
    because people mistype 'memroy' and abbreviate everything."""
    hits: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9/+-]*", text):
        word = m.group(0).lower()
        if len(word) < 2:
            continue
        best, best_score = None, 0.0
        for pid, words in vocab.items():
            for w in words:
                if word == w:
                    best, best_score = pid, 1.0
                    break
                # a fuzzy match cannot differ by much in length or start far
                # apart; checking that first is what keeps featurising cheap
                if abs(len(word) - len(w)) > 2 or (word[0] != w[0] and word[1:2] != w[1:2]):
                    continue
                score = difflib.SequenceMatcher(None, word, w).ratio()
                if score > best_score:
                    best, best_score = pid, score
            if best_score == 1.0:
                break
        if best is not None and best_score >= fuzzy:
            hits.append((m.start(), m.end(), best))
    # merge adjacent spans naming the same part ("cpu core", "io die")
    merged: List[Tuple[int, int, str]] = []
    for span in hits:
        if merged and span[2] == merged[-1][2] and span[0] - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], span[1], span[2])
        else:
            merged.append(span)
    return merged


def mask(text: str, vocab: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    """The clause with parts and numbers replaced by placeholders, and the
    parts it named, in order. This is what the model actually reads."""
    spans = find_parts(text, vocab)
    out, last, ids = [], 0, []
    for start, end, pid in spans:
        out.append(text[last:start])
        out.append("<part2>" if ids and pid != ids[0] else "<part>")
        ids.append(pid)
        last = end
    out.append(text[last:])
    masked = "".join(out)
    for token, pattern in UNITS:
        masked = pattern.sub(token, masked)
    return masked.lower(), ids


def features(masked: str, dim: int = DIM) -> np.ndarray:
    """Hashed word 1-2 grams and character 3-5 grams, L2 normalised."""
    vec = np.zeros(dim)
    words = ["^"] + re.findall(r"<[a-z0-9]+>|[a-z0-9'/+-]+", masked) + ["$"]
    grams = [f"w:{w}" for w in words]
    grams += [f"w:{a}_{b}" for a, b in zip(words, words[1:])]
    padded = f" {masked} "
    for n in (3, 4, 5):
        grams += [f"c:{padded[i:i + n]}" for i in range(len(padded) - n + 1)]
    for g in grams:
        # crc32, not hash(): Python randomises string hashing per process, so
        # hash() would put the same n-gram in a different bucket tomorrow and
        # a saved model would read its own weights as noise.
        vec[zlib.crc32(g.encode("utf-8")) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


@dataclass
class LanguageModel:
    weights: np.ndarray                      # (dim, classes)
    bias: np.ndarray                         # (classes,)
    classes: Tuple[str, ...]
    threshold: float = 0.5
    dim: int = DIM

    def scores(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights + self.bias
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def classify(self, masked: str) -> Tuple[str, float]:
        p = self.scores(features(masked, self.dim))
        i = int(np.argmax(p))
        return self.classes[i], float(p[i])

    def to_json(self) -> dict:
        return {"classes": list(self.classes), "dim": self.dim, "threshold": self.threshold,
                "weights": np.round(self.weights, 5).tolist(), "bias": np.round(self.bias, 5).tolist()}

    @staticmethod
    def from_json(d: dict) -> "LanguageModel":
        return LanguageModel(weights=np.asarray(d["weights"]), bias=np.asarray(d["bias"]),
                             classes=tuple(d["classes"]), threshold=float(d["threshold"]),
                             dim=int(d["dim"]))


@dataclass
class Read:
    """One clause, as the model read it."""
    text: str
    intent: str
    confidence: float
    parts: List[str]
    understood: bool


def segment(text: str) -> List[str]:
    """Split a request on the marks that always separate asks. Weaker
    boundaries -- "and", and a second ask with no connective at all -- are
    PROPOSED in read(), where the model can judge whether they are real."""
    pieces = [p.strip() for p in SPLIT.split(text) if p and p.strip()]
    return pieces or ([text.strip()] if text.strip() else [])


def second_ask_split(piece: str, vocab: Dict[str, List[str]]) -> List[str]:
    """"under 85 C at 30 W on a 20 mm package" is three asks in one breath.
    Cut before each preposition that introduces a quantity."""
    masked, _ids = mask(piece, vocab)
    cuts = [m.start() for m in SECOND_ASK.finditer(masked)]
    return _cut_like(piece, masked, cuts) if cuts else [piece]


def _cut_like(text: str, masked: str, cuts: Sequence[int]) -> List[str]:
    """Cut `text` where `masked` says, by counting words rather than
    characters: masking changes lengths but never word boundaries."""
    words_at = [len(masked[:c].split()) for c in cuts]
    words = text.split()
    pieces, last = [], 0
    for w in words_at:
        w = max(last + 1, min(w, len(words) - 1))
        pieces.append(" ".join(words[last:w]))
        last = w
    pieces.append(" ".join(words[last:]))
    return [p for p in pieces if p.strip()]


def read(text: str, parts: Sequence[dict], model: LanguageModel,
         merge_margin: float = 0.05) -> List[Read]:
    """Read a whole request: split it, classify the pieces, then let the model
    choose the segmentation.

    Splitting on connectives cuts some constraints in half, and the halves can
    look confident on their own -- "keep the cpu" reads as "keep it low", which
    is a real instruction. So the pieces are not final: any two neighbours are
    merged when the model reads the PAIR better than it reads them apart. That
    is what keeps "keep cpu and gpu adjacent" one constraint over two parts,
    and what folds a trailing "...the connectors are there" back into the ask
    it explains."""
    vocab = part_vocabulary(parts)

    def classify(piece: str) -> Read:
        masked, ids = mask(piece, vocab)
        intent, confidence = model.classify(masked)
        return Read(piece, intent, confidence, ids,
                    confidence >= model.threshold and intent != "NONE")

    def accept_split(pieces: List[str], whole: Read) -> Optional[List[Read]]:
        """A proposed split is taken only if the model reads the pieces as
        asks in their own right. A quantity settles it -- a number belongs to
        exactly one ask -- and a piece without one has to stand on its own
        confidently, because "keep the cpu" reads as an instruction ("keep it
        low") even when it is only half of one."""
        if len(pieces) < 2:
            return None
        reads_ = [classify(p) for p in pieces]
        quantities = [bool(re.search(r"<(?:temp|watt|mm)>", mask(p, vocab)[0])) for p in pieces]
        solid = all(r.understood and (q or r.confidence >= 0.75)
                    for r, q in zip(reads_, quantities))
        if any(quantities) and solid:
            return reads_
        apart = sum(r.confidence for r in reads_) / len(reads_)
        if all(r.understood for r in reads_) and apart > whole.confidence + 0.15:
            return reads_
        return None

    reads: List[Read] = []
    for piece in segment(text):
        whole = classify(piece)
        taken = accept_split(second_ask_split(piece, vocab), whole)
        if taken is not None:
            reads.extend(taken)
            continue
        halves = [h.strip() for h in WEAK.split(piece) if h.strip()]
        if len(halves) > 1:
            parts_read = [classify(h) for h in halves]
            apart = sum(r.confidence for r in parts_read) / len(parts_read)
            # A QUANTITY settles it: a number belongs to exactly one ask, so
            # "the dram on the left and nothing over 120 degrees" is two.
            # Without one, "and" is usually joining parts inside a single ask
            # -- "keep the cpu and gpu in the middle" -- and splitting it
            # leaves "keep the cpu", which reads as its own instruction. So
            # splitting then has to be clearly better, not better by a hair.
            taken = accept_split(halves, whole)
            if taken is not None:
                # each half may itself pack a second ask
                for half in taken:
                    inner = accept_split(second_ask_split(half.text, vocab), half)
                    reads.extend(inner if inner is not None else [half])
                continue
        reads.append(whole)

    # "put the cpu next to it": a pronoun means the part named just before.
    for i, r in enumerate(reads):
        if r.parts or not re.search(r"\b(?:it|that|this|them|those|there)\b", r.text, re.I):
            continue
        earlier = [p for prev in reads[:i] for p in prev.parts]
        if earlier:
            r.parts = [earlier[-1]]
    for i, r in enumerate(reads):
        if r.intent == "NEAR" and len(set(r.parts)) < 2:
            earlier = [p for prev in reads[:i] for p in prev.parts if p not in r.parts]
            if earlier and re.search(r"\b(?:it|that|this|them|those)\b", r.text, re.I):
                r.parts = list(dict.fromkeys(r.parts + [earlier[-1]]))
        r.understood = r.confidence >= model.threshold and r.intent != "NONE"
    # Only a FRAGMENT is merged -- a piece the model could not read as an ask
    # on its own. Merging two readable asks would throw one of them away,
    # which is what happens to "the hbm left, the io at the edge" if the rule
    # is "merge whenever the pair scores better".
    changed = True
    while changed and len(reads) > 1:
        changed = False
        for i, r in enumerate(reads):
            if r.understood:
                continue
            for j in (i - 1, i + 1):            # merge backwards, then forwards
                if not 0 <= j < len(reads):
                    continue
                a, b = (reads[j], r) if j < i else (r, reads[j])
                merged = classify(f"{a.text} and {b.text}")
                if merged.understood and merged.confidence >= reads[j].confidence:
                    lo, hi = min(i, j), max(i, j)
                    reads[lo:hi + 1] = [merged]
                    changed = True
                    break
            if changed:
                break
    for r in reads:                     # a constraint with no part is not actionable
        if r.understood and r.intent in NEEDS_PART and not r.parts:
            r.understood = False
        if r.understood and r.intent == "NEAR" and len(set(r.parts)) < 2:
            r.understood = False
    return reads


def load_registered(name: str = "copilot-language", version: Optional[int] = None
                    ) -> Tuple[LanguageModel, dict]:
    from ai import registry
    params, card = registry.load(name, version)
    return LanguageModel.from_json(params), card


NUMBER = {
    "temp": re.compile(r"(\d+(?:\.\d+)?)\s*" + DEG + r"\b", re.I),
    "watt": re.compile(r"(\d+(?:\.\d+)?)\s*" + WATTS + r"\b", re.I),
    "mm": re.compile(r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?\s*)?" + MM + r"\b", re.I),
}


def numbers(text: str) -> Dict[str, float]:
    """The quantities in a clause. Deterministic on purpose: a model should
    decide what a sentence ASKS for, not what 85 means."""
    out: Dict[str, float] = {}
    for kind, pattern in NUMBER.items():
        m = pattern.search(text)
        if m:
            out[kind] = float(m.group(1))
    if "temp" not in out and "watt" not in out and "mm" not in out:
        bare = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if bare:
            out["bare"] = float(bare.group(1))
    return out
