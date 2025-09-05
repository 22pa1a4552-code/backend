import re

# A simple keyword-based filter to allow only construction-related queries
CONSTRUCTION_KEYWORDS = [
    "concrete", "cement", "mortar", "aggregate", "brick", "steel",
    "beam", "column", "slab", "foundation", "masonry", "plaster",
    "construction", "building", "structural", "rebar", "reinforcement",
    "formwork", "shuttering", "estimation", "cost", "material",
    "safety", "scaffolding", "codes", "design", "load", "strength"
]


def is_construction_query(text: str):
    """
    Return (is_allowed, reason).
    Conservative: returns True only if there's a construction keyword.
    """
    lowered = text.lower()

    for kw in CONSTRUCTION_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            return True, f"Matched keyword: {kw}"

    return False, "No construction-related keywords found"
