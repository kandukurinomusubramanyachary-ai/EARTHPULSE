"""
EarthPulse — Feature 2: AI Query Interpreter.

Converts a free-text Earth question into a structured, executable analysis plan.

Design note
-----------
This is a deterministic, dependency-free semantic parser rather than an LLM
call. That is a deliberate MVP choice:

  * it runs offline, in ~1 ms, with no API key and no rate limit;
  * it is fully reproducible, which matters because the plan it emits decides
    which physics (spectral index) is applied to the pixels;
  * every decision is traceable to a matched phrase, which we surface in the
    UI as "why we read your question this way".

The intent vocabulary is expressed as concept -> synonym sets, which gives us
Feature 3 (semantic retrieval) for free: "deforestation", "tree cover loss"
and "forests disappearing" all collapse to the VEGETATION_LOSS concept.
The LLM upgrade path is documented in the TRD (§6.2): swap `interpret()` for a
function-calling model that emits the same `QueryPlan` schema.
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Concept vocabulary  (Feature 3 — semantic satellite search)
# ---------------------------------------------------------------------------
# Each analysis mode declares the phrases that map onto it. Longer phrases are
# matched first so "vegetation loss" beats a bare "vegetation".
CONCEPTS: dict[str, dict[str, Any]] = {
    "urban_expansion": {
        "label": "Urban / built-up expansion",
        "index": "NDBI",
        "direction": "increase",
        "icon": "building",
        "phrases": [
            "urban expansion", "urban growth", "urban sprawl", "urbanisation",
            "urbanization", "built up", "built-up", "builtup", "new construction",
            "construction", "new buildings", "new building", "buildings appeared",
            "concrete", "development", "developed", "real estate", "housing",
            "settlement", "settlements", "infrastructure", "layout", "township",
            "encroachment", "encroach", "roads", "road development", "impervious",
            "city expanded", "city growth", "expansion of the city",
        ],
    },
    "vegetation_loss": {
        "label": "Vegetation / forest loss",
        "index": "NDVI",
        "direction": "decrease",
        "icon": "tree",
        "phrases": [
            "vegetation loss", "vegetation decline", "loss of vegetation",
            "deforestation", "deforested", "forest loss", "forest cover loss",
            "tree cover loss", "tree loss", "forests disappearing",
            "forest disappeared", "forests are disappearing", "losing forest",
            "green cover loss", "green cover reduction", "vegetation removed",
            "cleared", "clearing", "logging", "canopy loss", "barren",
            "vegetation reduced", "greenery reduced", "loss of greenery",
        ],
    },
    "vegetation_gain": {
        "label": "Vegetation growth / regreening",
        "index": "NDVI",
        "direction": "increase",
        "icon": "sprout",
        "phrases": [
            "vegetation growth", "vegetation gain", "vegetation increased",
            "revegetation", "reforestation", "afforestation", "regreening",
            "greening", "green cover increase", "greener", "plantation",
            "crop growth", "more vegetation", "tree cover gain",
        ],
    },
    "water_loss": {
        "label": "Water-body shrinkage",
        "index": "MNDWI",
        "direction": "decrease",
        "icon": "droplet",
        "phrases": [
            "water loss", "lake shrinking", "lakes became smaller",
            "lake shrank", "shrinking", "shrunk", "drying", "dried up",
            "drying up", "water reduced", "water body reduced",
            "reduced in area", "reservoir level", "receding", "receded",
            "water level dropped", "drought", "desiccation", "smaller lake",
            "lost water", "water decline",
        ],
    },
    "water_gain": {
        "label": "Water expansion / flooding",
        "index": "MNDWI",
        "direction": "increase",
        "icon": "waves",
        "phrases": [
            "flood", "flooding", "flooded", "inundation", "inundated",
            "water expansion", "water expanded", "water increased",
            "submerged", "submergence", "overflow", "overflowed",
            "water spread", "deluge", "waterlogging", "water logging",
            "reservoir filled", "lake grew", "more water",
        ],
    },
    "burn_scar": {
        "label": "Burned area / fire scar",
        "index": "NBR",
        "direction": "decrease",
        "icon": "flame",
        "phrases": [
            "burn", "burned", "burnt", "burn scar", "fire", "wildfire",
            "forest fire", "bushfire", "fire damage", "scorched", "charred",
            "burned area", "fire affected",
        ],
    },
}

# Ordered longest-phrase-first for greedy, unambiguous matching.
_PHRASE_INDEX: list[tuple[str, str]] = sorted(
    ((p, mode) for mode, spec in CONCEPTS.items() for p in spec["phrases"]),
    key=lambda t: -len(t[0]),
)

# Words that signal the user wants a change analysis at all.
_CHANGE_CUES = [
    "change", "changed", "changing", "difference", "compare", "comparison",
    "before after", "over time", "trend", "evolution", "happened", "since",
    "between", "grew", "shrank", "expanded", "increase", "decrease", "loss",
    "gain", "appeared", "disappeared", "new",
]

_SIMILARITY_CUES = [
    "similar", "like this", "same pattern", "comparable", "resembl",
    "other places", "elsewhere", "find more",
]

# Feature words that constrain the analysis to a neighbourhood.
_PROXIMITY_FEATURES = {
    "water": ["lake", "lakes", "pond", "ponds", "reservoir", "river", "rivers",
              "canal", "tank", "tanks", "water body", "water bodies", "coast",
              "shoreline", "wetland", "wetlands"],
    "forest": ["forest", "forests", "woodland", "reserve", "sanctuary",
               "national park", "jungle"],
    "urban": ["city", "town", "urban area", "settlement"],
    "farmland": ["farmland", "farm", "farms", "agriculture", "agricultural",
                 "cropland", "fields", "crops"],
}

# Words that are location-ish noise and must not be treated as a place name.
_STOPWORDS = {
    "show", "me", "find", "where", "what", "which", "did", "does", "do", "has",
    "have", "had", "is", "are", "was", "were", "the", "a", "an", "in", "on",
    "at", "of", "for", "to", "from", "between", "and", "or", "near", "around",
    "close", "by", "over", "during", "since", "last", "past", "this", "that",
    "these", "those", "there", "here", "any", "all", "some", "new", "old",
    "please", "can", "you", "us", "our", "my", "his", "her", "its", "their",
    "area", "areas", "region", "regions", "place", "places", "zone", "zones",
    "district", "districts", "years", "year", "months", "month", "days",
    "day", "time", "period", "map", "image", "images", "satellite", "data",
    "analysis", "detect", "detection", "identify", "changes", "change",
    "happened", "occurred", "much", "many", "how", "why", "when",
    # concept words must never leak into the place name
    "urban", "expansion", "growth", "construction", "buildings", "building",
    "vegetation", "forest", "forests", "deforestation", "water", "lake",
    "lakes", "flood", "flooding", "loss", "gain", "cover", "green", "burn",
    "burned", "fire", "crop", "crops", "agriculture", "built", "up",
}

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Sentinel-2 has no usable archive before mid-2015.
S2_EPOCH_START = 2016


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
@dataclass
class QueryPlan:
    """Structured, executable interpretation of a natural-language question."""

    raw_query: str
    mode: str                       # key into CONCEPTS
    mode_label: str
    index: str                      # NDVI | NDBI | MNDWI | NBR
    direction: str                  # increase | decrease
    place: str | None
    start_year: int
    end_year: int
    proximity: str | None = None    # water | forest | urban | farmland
    wants_similarity: bool = False
    confidence: float = 0.0         # interpreter's own confidence, 0-1
    reasoning: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")   # en/em dash
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).strip()


def _extract_years(q: str, today: _dt.date) -> tuple[int, int, list[str], list[str]]:
    """Resolve the observation window. Returns (start, end, reasoning, assumptions)."""
    reasoning: list[str] = []
    assumptions: list[str] = []
    current_year = today.year

    # 1. Explicit range: "2022 to 2026", "2022-2026", "between 2022 and 2026"
    m = re.search(
        r"(19|20)(\d{2})\s*(?:-|to|and|until|till|through|vs\.?|versus)\s*(19|20)(\d{2})",
        q,
    )
    if m:
        a = int(m.group(1) + m.group(2))
        b = int(m.group(3) + m.group(4))
        lo, hi = min(a, b), max(a, b)
        reasoning.append(f"Explicit year range detected in the question: {lo}\u2013{hi}.")
        return lo, hi, reasoning, assumptions

    # 2. Relative window: "last 5 years", "past decade"
    m = re.search(r"(?:last|past|previous|recent)\s+(\d{1,2})\s*(year|yr)", q)
    if m:
        span = max(1, int(m.group(1)))
        reasoning.append(f"Relative window '{m.group(0)}' resolved against {current_year}.")
        return current_year - span, current_year, reasoning, assumptions

    if re.search(r"(?:last|past)\s+decade", q):
        reasoning.append(f"'Past decade' resolved against {current_year}.")
        return current_year - 10, current_year, reasoning, assumptions

    if re.search(r"(?:last|past)\s+(?:year|12\s*months)", q):
        reasoning.append("'Last year' resolved to a 1-year window.")
        return current_year - 1, current_year, reasoning, assumptions

    # 3. "since 2020"
    m = re.search(r"since\s+((?:19|20)\d{2})", q)
    if m:
        lo = int(m.group(1))
        reasoning.append(f"Open-ended window 'since {lo}' closed at the present year.")
        return lo, current_year, reasoning, assumptions

    # 4. "after the 2023 flood" / any two loose years
    years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", q)]
    years = [y for y in years if S2_EPOCH_START - 1 <= y <= current_year + 1]
    if len(years) >= 2:
        lo, hi = min(years), max(years)
        reasoning.append(f"Two years mentioned; interpreted as the window {lo}\u2013{hi}.")
        return lo, hi, reasoning, assumptions
    if len(years) == 1:
        y = years[0]
        if y >= current_year:
            reasoning.append(f"Single year {y} interpreted as a 2-year window ending now.")
            return max(S2_EPOCH_START, y - 2), current_year, reasoning, assumptions
        reasoning.append(f"Single year {y} interpreted as {y} \u2192 {current_year}.")
        return y, current_year, reasoning, assumptions

    # 5. Nothing stated — default to a 4-year window.
    lo, hi = current_year - 4, current_year
    assumptions.append(
        f"No time period was given, so a default {lo}\u2013{hi} window was used."
    )
    return lo, hi, reasoning, assumptions


def _extract_place(q_raw: str) -> tuple[str | None, list[str]]:
    """Pull the most plausible place name out of the question."""
    reasoning: list[str] = []
    q = q_raw

    # Strip a trailing/leading year range so it never joins the place string.
    q = re.sub(r"(?:between\s+)?(?:19|20)\d{2}\s*(?:-|to|and)\s*(?:19|20)\d{2}", " ", q)
    q = re.sub(r"\b(?:19|20)\d{2}\b", " ", q)
    q = re.sub(r"(?:last|past|previous|recent)\s+\d{1,2}\s*(?:year|yr)s?", " ", q)

    # Preposition-anchored capture: "in/near/around/at <Place>"
    # Comma-separated administrative chains must be captured WHOLE.
    #
    # Previously the pattern stopped at the first comma, so
    # "Terai, Nepal" -> "Terai", which Nominatim resolves to a village in
    # Madhya Pradesh, INDIA. The analysis then silently ran on the wrong
    # continent instead of failing. Keeping the qualifier is what
    # disambiguates duplicated place names worldwide.
    _TOK = r"(?:[A-Z][\w'\-]*|and|of|the|de|el|la|al)"
    prep = re.search(
        r"\b(?:in|near|around|at|for|within|inside|across|surrounding|close to)\s+"
        rf"({_TOK}(?:\s+{_TOK}){{0,3}}"           # first component
        rf"(?:\s*,\s*{_TOK}(?:\s+{_TOK}){{0,3}}){{0,4}})",  # ", Region, Country"
        q,
    )
    if prep:
        cand = _clean_place(prep.group(1))
        if cand:
            reasoning.append(f"Location '{cand}' captured after a locative preposition.")
            return cand, reasoning

    # Fallback: longest run of capitalised tokens not at sentence start only.
    tokens = q.split()
    best: list[str] = []
    cur: list[str] = []
    for i, tok in enumerate(tokens):
        bare = re.sub(r"[^\w'\-]", "", tok)
        if not bare:
            if len(cur) > len(best):
                best = cur
            cur = []
            continue
        is_cap = bare[:1].isupper()
        # ignore a capitalised first word ("Show me...") unless it continues
        if is_cap and not (i == 0 and bare.lower() in _STOPWORDS):
            if bare.lower() not in _STOPWORDS:
                cur.append(bare)
                continue
        if len(cur) > len(best):
            best = cur
        cur = []
    if len(cur) > len(best):
        best = cur

    if best:
        cand = _clean_place(" ".join(best))
        if cand:
            reasoning.append(f"Location '{cand}' inferred from capitalised place tokens.")
            return cand, reasoning

    # Last resort: lowercase gazetteer-ish scan after a preposition.
    prep2 = re.search(
        r"\b(?:in|near|around|at|for|within)\s+([a-z][\w'\-]*(?:\s+[a-z][\w'\-]*){0,2})", q
    )
    if prep2:
        cand = _clean_place(prep2.group(1))
        if cand:
            reasoning.append(f"Location '{cand}' inferred from a lowercase locative phrase.")
            return cand, reasoning

    return None, reasoning


def _clean_place(s: str) -> str | None:
    words = [w for w in re.split(r"\s+", s.strip()) if w]
    while words and words[0].lower() in _STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in _STOPWORDS:
        words.pop()
    if not words:
        return None
    out = " ".join(words).strip(" ,.;:!?")
    return out if len(out) >= 3 else None


def _match_concept(q: str) -> tuple[str | None, list[str]]:
    """Greedy longest-phrase concept match. Returns (mode, matched phrases)."""
    hits: list[tuple[str, str]] = []
    consumed = q
    for phrase, mode in _PHRASE_INDEX:
        if phrase in consumed:
            hits.append((phrase, mode))
            consumed = consumed.replace(phrase, " ")
    if not hits:
        return None, []

    # Score each mode by total matched phrase length (longer = more specific).
    scores: dict[str, int] = {}
    for phrase, mode in hits:
        scores[mode] = scores.get(mode, 0) + len(phrase)
    best_mode = max(scores, key=lambda m: scores[m])
    matched = [p for p, m in hits if m == best_mode]
    return best_mode, matched


        # Subject nouns used by the compositional fallback parser.
_SUBJECTS = {
    "water": ["water bod", "water bodies", "waterbody", "lake", "lakes", "pond",
              "reservoir", "tank", "river", "wetland", "water level", "water"],
    "vegetation": ["vegetation", "forest", "tree", "canopy", "green cover",
                   "greenery", "woodland", "jungle", "crop", "plantation"],
    "urban": ["building", "construction", "built", "city", "town", "settlement",
              "infrastructure", "road", "concrete"],
}

_DECREASE_VERBS = [
    "reduc", "shrink", "shrank", "shrunk", "declin", "decreas", "loss", "lost",
    "disappear", "dry", "dried", "smaller", "less", "fell", "drop", "receded",
    "vanish", "degrad", "depleted", "diminish",
]
_INCREASE_VERBS = [
    "increas", "expand", "grew", "grow", "gain", "more", "rise", "risen",
    "appear", "new", "spread", "larger", "bigger", "up",
]


def _subject_fallback(q: str) -> tuple[str | None, list[str]]:
    """
    Compositional fallback: <subject noun> + <direction verb> -> mode.

    Handles phrasings the phrase table cannot enumerate, e.g.
    "which water bodies have significantly reduced" or
    "has the greenery around here declined".
    """
    subject = None
    matched: list[str] = []
    for key, nouns in _SUBJECTS.items():
        for n in nouns:
            if n in q:
                subject, matched = key, [n]
                break
        if subject:
            break
    if not subject:
        return None, []

    def _first_at(verbs: list[str]) -> int:
        hits = [q.find(v) for v in verbs if v in q]
        return min(hits) if hits else 10**6

    d_at, u_at = _first_at(_DECREASE_VERBS), _first_at(_INCREASE_VERBS)
    down, up = d_at < 10**6, u_at < 10**6

    if down and up:
        # Both directions stated ("did water expand or shrink?"). Resolve
        # deterministically to whichever the user mentioned first.
        down, up = (d_at < u_at), (u_at < d_at)
    elif not down and not up:
        if subject == "urban":
            return "urban_expansion", matched
        return None, []

    mapping = {
        ("water", True): "water_loss",
        ("water", False): "water_gain",
        ("vegetation", True): "vegetation_loss",
        ("vegetation", False): "vegetation_gain",
        ("urban", True): "urban_expansion",   # built-up is analysed as increase
        ("urban", False): "urban_expansion",
    }
    return mapping.get((subject, down)), matched


def _detect_proximity(q: str) -> str | None:
    """Detect 'near <feature>' constraints, e.g. 'construction near lakes'."""
    if not re.search(r"\b(near|around|beside|adjacent|next to|along|close to|by the)\b", q):
        return None
    for key, words in _PROXIMITY_FEATURES.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", q):
                return key
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def interpret(
    query: str,
    *,
    today: _dt.date | None = None,
    place_override: str | None = None,
    start_override: int | None = None,
    end_override: int | None = None,
) -> QueryPlan:
    """Parse a natural-language Earth question into an executable plan."""
    today = today or _dt.date.today()
    raw = _normalise(query or "")
    q = raw.lower()

    reasoning: list[str] = []
    assumptions: list[str] = []
    confidence = 0.35

    # --- 1. Phenomenon ----------------------------------------------------
    mode, matched = _match_concept(q)

    # A bare subject noun ("water bodies") must not outrank an explicit
    # directional statement ("...have significantly reduced"). If the phrase
    # table only matched a generic term, let the compositional parser refine it.
    generic_only = bool(matched) and all(len(m) <= 6 for m in matched)
    if mode is None or generic_only:
        sub_mode, sub_matched = _subject_fallback(q)
        if sub_mode and sub_mode != mode:
            mode = sub_mode
            matched = (matched + sub_matched) if generic_only else sub_matched
            reasoning.append(
                "Interpreted from sentence structure: the subject of the question combined "
                "with a directional verb determined the analysis type."
            )

    if mode:
        confidence += 0.30
        pretty = ", ".join(f"'{m}'" for m in matched[:3])
        reasoning.append(
            f"Matched {pretty} \u2192 {CONCEPTS[mode]['label']}, "
            f"analysed with the {CONCEPTS[mode]['index']} spectral index."
        )
    else:
        mode = "urban_expansion"
        assumptions.append(
            "No specific phenomenon was recognised in the question; defaulted to "
            "built-up change detection. Rephrase with words like 'vegetation loss', "
            "'flooding' or 'lake shrinking' to switch the analysis."
        )

    spec = CONCEPTS[mode]

    # Directional override: "where did vegetation increase" flips the sign.
    direction = spec["direction"]
    if mode in ("vegetation_loss", "vegetation_gain"):
        if re.search(r"\b(increase|increased|growth|grew|gain|gained|more|greener)\b", q):
            mode, direction = "vegetation_gain", "increase"
        elif re.search(r"\b(decrease|decreased|loss|lost|less|reduced|declin)\w*\b", q):
            mode, direction = "vegetation_loss", "decrease"
        spec = CONCEPTS[mode]

    # --- 2. Change intent -------------------------------------------------
    if any(cue in q for cue in _CHANGE_CUES):
        confidence += 0.10
        reasoning.append("Question implies a temporal comparison, so multi-temporal change detection was selected.")
    else:
        assumptions.append("The question did not explicitly request a comparison; change detection was applied by default.")

    # --- 3. Location ------------------------------------------------------
    if place_override:
        place = place_override.strip()
        reasoning.append(f"Location '{place}' supplied directly by the user.")
        confidence += 0.20
    else:
        place, place_reason = _extract_place(raw)
        reasoning.extend(place_reason)
        if place:
            confidence += 0.20
        else:
            assumptions.append("No location could be extracted from the question.")

    # --- 4. Time ----------------------------------------------------------
    if start_override and end_override:
        start_year, end_year = int(start_override), int(end_override)
        reasoning.append(f"Observation window {start_year}\u2013{end_year} set directly by the user.")
        confidence += 0.10
    else:
        start_year, end_year, tr, ta = _extract_years(q, today)
        reasoning.extend(tr)
        assumptions.extend(ta)
        if tr:
            confidence += 0.10

    # Clamp to the Sentinel-2 archive.
    if start_year < S2_EPOCH_START:
        assumptions.append(
            f"Sentinel-2 imagery only begins in {S2_EPOCH_START}; the start of the "
            f"window was moved from {start_year} to {S2_EPOCH_START}."
        )
        start_year = S2_EPOCH_START
    if end_year > today.year:
        end_year = today.year
    if end_year <= start_year:
        end_year = min(today.year, start_year + 1)

    # --- 5. Modifiers -----------------------------------------------------
    proximity = _detect_proximity(q)
    if proximity:
        reasoning.append(
            f"Spatial constraint detected: results will be highlighted near {proximity} features."
        )
    wants_similarity = any(c in q for c in _SIMILARITY_CUES)
    if wants_similarity:
        reasoning.append("Question asks for comparable locations, so similarity search was enabled.")

    return QueryPlan(
        raw_query=raw,
        mode=mode,
        mode_label=spec["label"],
        index=spec["index"],
        direction=direction,
        place=place,
        start_year=start_year,
        end_year=end_year,
        proximity=proximity,
        wants_similarity=wants_similarity,
        confidence=round(min(confidence, 0.98), 2),
        reasoning=reasoning,
        matched_terms=matched,
        assumptions=assumptions,
    )
