"""
ai/language_corpus.py
ThermalForge :: the training corpus for the copilot's language model.

A grammar only understands the phrasings somebody thought of. A model can
generalise past them -- but only if what it trains on is varied enough to
teach phrasing rather than keywords. So this generates requests the way people
actually write them: different verbs, different word order, contractions,
politeness, filler, abbreviations, units written five ways, and typos.

Every example is one CLAUSE with one intent. Whole requests are built by
joining clauses with the connectives people use, because the parser has to
find the clause boundaries too.

What this is NOT is the measurement set. Scoring a model on text from the same
templates that trained it measures memorisation of those templates. The honest
set is validation/language_heldout.json: sentences written by hand, none of
them produced by anything here, including ones that mean nothing to the app.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

# The intents the copilot can act on. NONE is not a failure: it is the answer
# for a clause that asks for something the app cannot do, and saying so is
# better than applying the nearest constraint that happens to match a keyword.
INTENTS = (
    "LIMIT", "TOTAL_POWER", "PART_POWER",
    "LEFT", "RIGHT", "TOP", "BOTTOM", "EDGE", "CORNER", "CENTRE",
    "FIXED", "NEAR",
    "COOLING_DIRECT", "COOLING_LID", "COOLING_HEATSINK", "COOLING_COLDPLATE",
    "SIZE", "OPTIMIZE", "NONE",
)

PART = "<part>"
PART2 = "<part2>"
TEMP = "<temp>"
WATT = "<watt>"
MM = "<mm>"

# Ways people write the same request. {p} is a part, {t} a temperature, {w} a
# power, {m} a length -- filled in by the generator with realistic surface
# forms, so the model sees "85C", "85 °C", "85 degrees" and "85c" alike.
TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "LIMIT": (
        "anything past {n} and we fail", "anything over {n} is a fail",
        "nothing past {n}", "we fail above {n}", "keep it under {n}",
        "under {n}", "below {n}", "no hotter than {n}", "{n} is the ceiling",
        "hold the hot spot to {t}", "hold it to {t}", "it cannot get hotter than {t}",
        "cannot get hotter than {t}", "must stay cooler than {t}",
        "what does it peak at against {t}", "will it stay under {t}",
        "does it stay below {t}", "can it hold {t}", "is it under {t}",
        "it must not go over {t}", "we cannot go above {t}", "spec is {t}",
        "spec'd to {t}", "rated to {t}", "the requirement is {t}",
        "{t} is the number", "{t} is the ceiling", "anything over {t} is a fail",
        "under {t}", "below {t}", "keep it under {t}", "keep it below {t}",
        "stay under {t}", "stay below {t}", "no hotter than {t}", "no more than {t}",
        "not above {t}", "never above {t}", "nothing above {t}", "must not exceed {t}",
        "it cannot exceed {t}", "the peak must stay under {t}", "peak below {t}",
        "max {t}", "maximum {t}", "max temp {t}", "max temperature {t}",
        "limit is {t}", "the limit is {t}", "a limit of {t}", "hard limit {t}",
        "i need it under {t}", "we need it below {t}", "it has to stay under {t}",
        "make sure it stays below {t}", "keep the hot spot under {t}",
        "keep the junction below {t}", "junction under {t}", "hotspot under {t}",
        "the die must stay under {t}", "nothing hotter than {t}",
        "can you keep it under {t}", "please keep it below {t}",
        "target {t}", "budget of {t}", "thermal budget {t}", "ceiling of {t}",
        "cap it at {t}", "cap the peak at {t}", "at most {t}", "{t} or cooler",
        "{t} max", "{t} at the most", "under {t} please", "needs to be under {t}",
        "aim for under {t}", "keep temperatures below {t}", "keep temps under {t}",
    ),
    "TOTAL_POWER": (
        "what happens at {w}", "how hot at {w}", "what about {w}",
        "try {w}", "try it at {w}", "say {w}", "call it {w}", "roughly {w}",
        "about {w}", "around {w}", "if it draws {w}", "when it pulls {w}",
        "under a {w} load", "a {w} workload", "at {w} in total",
        "at {w}", "at {w} total", "{w} total", "total power {w}", "total of {w}",
        "running {w}", "running at {w}", "with {w}", "for {w}", "at a total of {w}",
        "the package draws {w}", "it draws {w}", "power budget {w}", "budget {w}",
        "when the whole thing draws {w}", "assume {w}", "assuming {w}",
        "scale the power to {w}", "set total power to {w}", "make the total {w}",
        "all up it is {w}", "{w} across the package", "{w} across all parts",
        "whole package at {w}", "under load at {w}", "at full load {w}",
    ),
    "PART_POWER": (
        "{p} alone is {w}", "{p} on its own is {w}", "bump {p} to {w}",
        "raise {p} to {w}", "drop {p} to {w}", "{p} burns {w}", "{p} eats {w}",
        "give the {p} {w}", "{p} at about {w}",
        "{p} at {w}", "{p} draws {w}", "{p} is {w}", "{p} runs at {w}",
        "set {p} to {w}", "give {p} {w}", "make {p} {w}", "{p} pulls {w}",
        "the {p} takes {w}", "with {p} at {w}", "{p} dissipates {w}",
        "put {p} at {w}", "{p} should be {w}", "{p} needs {w}",
    ),
    "LEFT": (
        "{p} on the left of the package", "shift {p} left",
        "{p} to the left side", "{p} sits left", "{p} left of centre",
        "{p} on the left", "{p} to the left", "put {p} on the left",
        "place {p} on the left", "keep {p} on the left", "move {p} to the left",
        "{p} on the left side", "{p} on the left hand side", "{p} left",
        "i want {p} on the left", "{p} should sit on the left",
        "{p} goes on the left", "{p} belongs on the left", "leftmost {p}",
        "{p} over on the left", "{p} on the west side", "{p} left side please",
        "can you put {p} on the left", "{p} needs to be on the left",
        "{p} and {p2} on the left", "{p} stays left",
    ),
    "RIGHT": (
        "{p} on the right of the package", "{p} belongs right",
        "{p} over to the right", "shift {p} right", "{p} to the right side",
        "{p} on the right", "put {p} on the right", "place {p} to the right",
        "{p} on the right side", "{p} right", "move {p} right",
        "{p} should be on the right", "{p} over on the right", "rightmost {p}",
        "{p} on the east side", "keep {p} on the right", "{p} goes right",
        "{p} and {p2} on the right",
    ),
    "TOP": (
        "{p} and {p2} at the top", "keep {p} and {p2} up top",
        "{p} at the top of the package", "{p} up at the top of the die",
        "{p} at the top", "{p} on top", "put {p} at the top", "{p} up top",
        "{p} towards the top", "{p} in the upper half", "{p} at the north end",
        "keep {p} at the top", "{p} should be near the top", "move {p} up",
        "{p} on the upper side",
    ),
    "BOTTOM": (
        "{p} and {p2} at the bottom", "keep {p} and {p2} low",
        "{p} at the bottom of the package", "{p} down at the bottom of the die",
        "{p} at the bottom", "put {p} at the bottom", "{p} down the bottom",
        "{p} in the lower half", "{p} at the south end", "move {p} down",
        "{p} towards the bottom", "keep {p} low", "{p} on the lower side",
    ),
    "EDGE": (
        "{p} and {p2} at the edge", "keep {p} and {p2} on the perimeter",
        "{p} has to reach the boundary", "{p} needs to reach the edge",
        "{p} where the connectors are", "{p} out by the pins",
        "{p} against the outside wall", "{p} at the rim", "{p} on the border",
        "{p} at the edge", "{p} on the edge", "put {p} at the edge",
        "{p} against the edge", "{p} near the edge", "{p} by the edge",
        "{p} on the perimeter", "{p} at the boundary", "{p} around the outside",
        "keep {p} at the edge", "{p} should touch the edge", "{p} out at the edge",
        "{p} on the outside", "{p} at the package edge", "{p} along the edge",
        "{p} needs to be at the edge for the connectors",
    ),
    "CORNER": (
        "{p} and {p2} in the corners", "keep {p} and {p2} cornered",
        "{p} in the corner of the package", "{p} in a corner of the die",
        "{p} cornered on the substrate",
        "{p} in the corner", "put {p} in a corner", "{p} in one of the corners",
        "{p} at a corner", "{p} tucked in the corner", "keep {p} in the corner",
        "{p} should be cornered", "{p} in the far corner",
    ),
    "CENTRE": (
        "{p} and {p2} in the centre", "keep {p} and {p2} in the middle",
        "{p} and {p2} centred", "put {p} and {p2} in the middle",
        "centre {p} on the die", "center {p} on the die", "{p} in the middle of the die",
        "{p} in the centre of the package", "{p} at the centre of the substrate",
        "{p} middle of the floorplan", "put {p} centrally on the die",
        "{p} in the centre", "{p} in the center", "{p} in the middle",
        "put {p} in the middle", "{p} centred", "{p} centered", "centre {p}",
        "center {p}", "{p} dead centre", "keep {p} in the middle",
        "{p} right in the centre", "{p} should be central", "{p} in the core",
    ),
    "FIXED": (
        "keep {p} and {p2} where they are", "{p} and {p2} are both fixed",
        "{p} position is fixed", "{p} position is frozen", "{p} position is locked",
        "{p} is frozen", "{p} is anchored", "{p} is bonded already",
        "{p} is already placed and stays", "{p} cannot move", "{p} must stay there",
        "freeze {p}", "freeze the {p} where it is", "{p} is committed",
        "{p} is locked by the socket", "{p} is set by the board",
        "keep {p} where it is", "leave {p} where it is", "don't move {p}",
        "do not move {p}", "{p} stays put", "{p} is fixed", "{p} must not move",
        "pin {p}", "{p} stays where it is", "hold {p} in place", "lock {p}",
        "{p} is already placed", "leave {p} alone", "{p} cannot be moved",
        "{p} is fixed by the package", "keep {p} in place",
    ),
    "NEAR": (
        "{p} side by side with {p2}", "{p} and {p2} side by side",
        "the shortest link from {p} to {p2}", "shortest possible link from {p} to {p2}",
        "{p} right beside {p2}", "{p} touching {p2} if you can",
        "short traces between {p} and {p2}", "short wires between {p} and {p2}",
        "keep {p} and {p2} adjacent", "{p} and {p2} side by side",
        "{p} hard up against {p2}", "minimise the distance between {p} and {p2}",
        "{p} wants to be close to {p2}", "as short a link as possible from {p} to {p2}",
        "{p} next to {p2}", "{p} near {p2}", "put {p} next to {p2}",
        "{p} beside {p2}", "{p} close to {p2}", "keep {p} near {p2}",
        "{p} adjacent to {p2}", "{p} should be near {p2}", "{p} by {p2}",
        "{p} right next to {p2}", "{p} as close as possible to {p2}",
        "short path between {p} and {p2}", "{p} and {p2} together",
        "{p} touching {p2}", "keep {p} and {p2} close",
    ),
    "COOLING_DIRECT": (
        "there is no lid", "there's no lid on it", "without a lid",
        "no heat spreader on this one", "no ihs", "we are not using a lid",
        "it ships without a lid", "lidless", "no spreader",
        "bare die", "no lid", "direct cooling", "coolant straight on the die",
        "bare die cooling", "leave it bare", "no heat spreader", "uncovered die",
        "direct to the silicon", "nothing on top of the die",
    ),
    "COOLING_LID": (
        "integrated heat spreader", "with an ihs", "adding a heat spreader",
        "it has a spreader", "we are adding a lid", "lid on top",
        "assume a lid", "spreader fitted",
        "with a lid", "add a lid", "put a lid on it", "use a lid",
        "lidded package", "with a heat spreader", "add an ihs", "with the lid on",
        "copper lid", "give it a lid",
    ),
    "COOLING_HEATSINK": (
        "air cooled", "air cooling", "finned heatsink", "with fins",
        "a heatsink with airflow", "fan and fins", "convection cooled with a sink",
        "decent heat sink", "big heatsink",
        "with a heatsink", "add a heat sink", "use a heatsink", "stick a heatsink on",
        "heatsink and fan", "with a fan and heatsink", "air cooled with a heatsink",
        "bolt on a heat sink", "forced air with a heatsink",
    ),
    "COOLING_COLDPLATE": (
        "on a cold plate", "cold plate", "coldplate", "water cooling",
        "liquid loop", "water block", "we are going liquid",
        "with a cold plate", "use a coldplate", "liquid cooling", "water cooled",
        "liquid cooled", "put it on a cold plate", "add a cold plate",
        "switch to liquid", "cold plate cooling", "direct liquid cooling",
    ),
    "SIZE": (
        "try it on a {m} substrate", "what about {m}", "how about a {m} package",
        "can we fit {m}", "move to a {m} package", "{m} substrate",
        "on a {m} package", "{m} package", "make the package {m}",
        "use a {m} substrate", "package size {m}", "{m} by {m} package",
        "shrink it to {m}", "grow the package to {m}", "a {m} part",
        "fit it on {m}", "{m} footprint", "resize the package to {m}",
    ),
    "OPTIMIZE": (
        "shuffle the chiplets", "work out the floorplan", "sort the layout out",
        "place everything sensibly", "best arrangement please", "where should things go",
        "lay the parts out", "do the floorplan",
        "optimise the layout", "optimize it", "find the coolest layout",
        "arrange the parts", "place everything", "lay it out", "rearrange it",
        "find the best placement", "sort out the floorplan", "do the placement",
        "give me the coolest arrangement", "move things around to cool it down",
        "what is the best layout", "improve the layout", "re-place the chiplets",
    ),
    "NONE": (
        "make it purple", "make it pretty", "add a logo", "send it to manufacturing",
        "what is the weather", "order pizza", "make it cheaper", "reduce the cost",
        "use a different vendor", "what time is it", "write me a poem",
        "make it quieter", "reduce the noise", "add more memory bandwidth",
        "increase the clock speed", "run at 3 ghz", "use tsmc n3",
        "make it look nicer", "change the colour", "email the result",
        "print the report", "explain quantum computing", "what should i eat",
        "add bluetooth", "make the fan quieter", "use less silicon area for cost",
        "tell me a joke", "who designed this", "schedule a meeting",
        "make it faster", "improve performance", "double the cores",
        "sign the package off", "book the fab slot", "talk to the vendor",
        # fragments: what splitting a real sentence leaves behind. Without
        # these the model labels "hard requirement" or "if you can" as an ask.
        "hard requirement", "if possible", "if you can", "please", "thanks",
        "for me", "obviously", "ideally", "ok", "right", "so", "well",
        "that's it", "the package", "this one", "in this case", "as before",
        "same as last time", "like i said", "you know", "anyway", "of course",
        "no problem", "sounds good", "the usual", "as discussed", "per spec",
        "it is what it is", "to be safe", "just in case", "for now",
        "at some point", "when you can", "quickly", "roughly", "more or less",
        "no exceptions", "no excuses", "no argument", "no debate", "full stop",
        "a cheaper substrate", "a different substrate material", "cheaper materials",
        "a cheaper package", "better yield", "what is the yield", "which node",
        "a different vendor", "second source it", "more bandwidth", "wider bus",
        "error correction", "more cache", "a faster link", "lower latency",
        # off-topic asks that carry NUMBERS, with units this app does not
        # model. Without these, any number in a sentence looks like a limit.
        "increase the bus width to 512 bits", "run it at 3.2 ghz",
        "clock it at 2800 mhz", "give it 16 lanes", "4 memory channels",
        "we need 900 gb/s", "2 tb of storage", "move to the 3 nm node",
        "12 metal layers", "1.1 volts on the core rail", "budget is 40 dollars",
        "8 more cores", "double the threads", "a 256 bit interface",
        "raise the voltage to 1.05 v", "50 percent more bandwidth",
        "who signed this off", "who approved the floorplan", "who owns this block",
        # the reason someone gives for an ask, trailing after it. It explains
        # a constraint; it is not a second one.
        "the connectors are there", "that is where the pins are",
        "that's where the cables go", "the board needs it there",
        "for signal integrity", "for the routing", "because of the socket",
        "it is a mechanical constraint", "the customer asked for it",
        "that is how it is bonded", "the package demands it", "for assembly",
        "to keep the traces short", "for the airflow", "that is the plan",
    ),
}

# Fillers people put in front of a request, and at the end.
PREFIXES = ("", "", "", "", "please ", "can you ", "could you ", "i want ", "i need ",
            "we need ", "let's have ", "make sure ", "try to ", "ideally ", "if possible ",
            "just ", "basically ", "so ", "ok ", "right, ")
SUFFIXES = ("", "", "", "", "", " please", " thanks", " if you can", " ideally",
            " for me", " obviously", " really", " at least")

TEMP_FORMS = ("{n}C", "{n} C", "{n}°C", "{n} °C", "{n} degrees", "{n} deg", "{n} degrees c",
              "{n} celsius", "{n}c", "{n} degc")
WATT_FORMS = ("{n}W", "{n} W", "{n} watts", "{n}w", "{n} watt", "{n} watts total")
MM_FORMS = ("{n}mm", "{n} mm", "{n} millimetre", "{n} millimeter", "{n}x{n}mm",
            "{n} x {n} mm", "{n} by {n} mm")

CONNECTIVES = (", ", ", and ", " and ", "; ", ", then ", " then ", ", also ", " also ",
               ". ", ", plus ", " while ", ", with ")


def _bare(rng: random.Random) -> str:
    return str(rng.choice([60, 70, 75, 80, 85, 90, 92, 95, 100, 105, 110]))


def _number(rng: random.Random, kind: str) -> str:
    if kind == "temp":
        n = rng.choice([60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 72.5, 87, 83])
        return rng.choice(TEMP_FORMS).format(n=n)
    if kind == "watt":
        n = rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 7.5, 13.5])
        return rng.choice(WATT_FORMS).format(n=n)
    n = rng.choice([10, 15, 20, 25, 30, 35, 40])
    return rng.choice(MM_FORMS).format(n=n)


def _typo(rng: random.Random, text: str) -> str:
    """People type badly. A model that only ever saw clean text learns to be
    brittle exactly where a user is most frustrated."""
    words = text.split()
    if not words:
        return text
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) < 4:
        return text
    kind = rng.random()
    j = rng.randrange(len(w) - 1)
    if kind < 0.35:                                    # swap two letters
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif kind < 0.6:                                   # drop a letter
        w = w[:j] + w[j + 1:]
    elif kind < 0.8:                                   # double a letter
        w = w[:j] + w[j] + w[j:]
    else:                                              # neighbouring key
        w = w[:j] + rng.choice("asdfghjkl") + w[j + 1:]
    words[i] = w
    return " ".join(words)


INTENSIFIERS = ("right ", "bang ", "smack ", "exactly ", "just ", "pretty much ",
                "more or less ", "well ", "way ")


def _intensify(rng: random.Random, text: str) -> str:
    """People stress where something goes: "bang in the middle", "right at the
    edge". Inserting them at random teaches every placement phrasing at once."""
    for preposition in (" in the ", " at the ", " on the ", " next to ", " against the "):
        if preposition in text and rng.random() < 0.5:
            return text.replace(preposition, " " + rng.choice(INTENSIFIERS).strip()
                                + preposition, 1)
    return text


def clause(rng: random.Random, intent: str, parts: Sequence[str],
           typo_rate: float = 0.12) -> str:
    template = rng.choice(TEMPLATES[intent])
    a, b = rng.sample(list(parts), 2) if len(parts) > 1 else (parts[0], parts[0])
    text = template.format(p=a, p2=b, t=_number(rng, "temp"), w=_number(rng, "watt"),
                           m=_number(rng, "mm"), n=_bare(rng))
    text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
    if rng.random() < 0.25:
        text = _intensify(rng, text)
    if rng.random() < typo_rate:
        text = _typo(rng, text)
    if rng.random() < 0.12:
        text = text.upper() if rng.random() < 0.3 else text.capitalize()
    return text.strip()


def request(rng: random.Random, parts: Sequence[str], n_clauses: int = 0
            ) -> Tuple[str, List[Tuple[str, str]]]:
    """A whole request and its clauses, so the parser learns boundaries too."""
    n = n_clauses or rng.choices([1, 2, 3, 4], weights=[3, 4, 3, 1])[0]
    intents = [rng.choices(INTENTS, weights=[3 if i != "NONE" else 4
                                             for i in INTENTS])[0] for _ in range(n)]
    pieces = [clause(rng, i, parts) for i in intents]
    text = pieces[0]
    for piece in pieces[1:]:
        text += rng.choice(CONNECTIVES) + piece
    return text, list(zip(pieces, intents))


def corpus(n: int, seed: int = 0, parts: Sequence[str] = ("CPU", "GPU", "memory", "SerDes",
                                                          "NPU", "HBM", "accelerator", "io die",
                                                          "cache", "the cpu core")
           ) -> List[Tuple[str, str]]:
    """n labelled CLAUSES, drawn through whole requests."""
    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    while len(out) < n:
        _text, labelled = request(rng, parts)
        out.extend(labelled)
    return out[:n]
