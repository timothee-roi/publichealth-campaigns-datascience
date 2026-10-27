"""Shared configuration: paths, campaigns, seasonal windows and queries.

Every notebook imports this module, so a change here (for example a new
query or a different window) propagates to the whole pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root, whatever the notebook's cwd is)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DAILY_DIR = DATA_PROCESSED / "daily"          # inputs of the Granger analysis
MONTHLY_DIR = DATA_PROCESSED / "monthly"      # long-run GT / MC series
QUERY_DIR = DATA_PROCESSED / "queries"        # snowball terms, k-grams
MEDICAM_RAW = DATA_RAW / "medicam"            # Medic'AM .xls files
MEDICAM_DIR = DATA_PROCESSED / "medicam"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"

for _d in (DATA_RAW, DAILY_DIR, MONTHLY_DIR, QUERY_DIR, MEDICAM_RAW,
           MEDICAM_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
GEO = "FR"
FIRST_YEAR, LAST_YEAR = 2015, 2025          # first and last campaign season
MONTHLY_TIMEFRAME = "2015-01-01 2026-01-01"

# MediaCloud: the API key is read from the environment, never hard-coded.
#   export MEDIACLOUD_API_KEY="..."
MEDIACLOUD_API_KEY = os.environ.get("MEDIACLOUD_API_KEY")
MC_FRENCH_NATIONAL_COLLECTION = 34412176


@dataclass(frozen=True)
class Campaign:
    code: str                 # short code used in file names
    label: str
    color: str
    # Daily window around each yearly campaign: (month, day) of start and end.
    # If the end comes before the start in the calendar, the window crosses
    # New Year (Dry January).
    window_start: tuple[int, int]
    window_end: tuple[int, int]
    # Composite Google Trends query (terms joined with " + "), as used to
    # produce data/processed/daily/GT_<code>_Granger.csv.
    gt_terms: list[str] = field(default_factory=list)
    # Optimised MediaCloud query (terms joined with OR), from the k-gram
    # selection (paper, Appendix A). The MC files in data/processed/daily/ do
    # not record the query that produced them, so check these terms against
    # your collection notes before re-collecting.
    mc_terms: list[str] = field(default_factory=list)
    # Snowball seeds and anchors for Google Trends query discovery.
    gt_seeds: list[str] = field(default_factory=list)
    gt_exclusions: list[str] = field(default_factory=list)

    @property
    def crosses_new_year(self) -> bool:
        return self.window_end < self.window_start

    def season_windows(self, first=FIRST_YEAR, last=LAST_YEAR):
        """Yield (season_label, start_date, end_date) for each yearly window."""
        import datetime as dt
        for y in range(first, last + 1):
            start = dt.date(y, *self.window_start)
            end_year = y + 1 if self.crosses_new_year else y
            end = dt.date(end_year, *self.window_end)
            label = f"{y}-{end_year}" if self.crosses_new_year else str(y)
            yield label, start, end


CAMPAIGNS: dict[str, Campaign] = {
    "OR": Campaign(
        code="OR", label="Octobre Rose", color="#d6457a",
        window_start=(9, 1), window_end=(11, 30),
        gt_terms=["Octobre rose", "cancer du sein", "mammographie",
                  "dépistage cancer du sein", "cancer du sein symptôme"],
        mc_terms=["octobre rose", "cancer sein", "lutte contre cancer sein",
                  "dépistage cancer sein", "mois octobre rose"],
        gt_seeds=["Octobre rose", "cancer du sein", "mammographie", "depistage sein"],
        gt_exclusions=[r"\brobe\b", r"\bmode\b", r"deco"],
    ),
    "Mov": Campaign(
        code="Mov", label="Movember", color="#6c4fb3",
        window_start=(10, 1), window_end=(12, 31),
        gt_terms=["Movember", "cancer prostate", "movember prostate",
                  "cancer prostate psa", "cancer prostate traitement"],
        mc_terms=["movember", "cancer prostate", "mois movember",
                  "masculin cancer prostate", "maladie masculin cancer prostate"],
        gt_seeds=["Movember", "cancer prostate", "depistage prostate", "PSA test"],
        gt_exclusions=[r"\brasoir\b", r"coiffeur"],
    ),
    "MB": Campaign(
        code="MB", label="Mars Bleu", color="#1f8a8a",
        window_start=(2, 2), window_end=(4, 30),
        gt_terms=["Mars bleu", "cancer colorectal", "dépistage colon",
                  "coloscopie", "cancer colon"],
        mc_terms=["mars bleu", "cancer colorectal", "dépistage cancer colorectal",
                  "cancer colorectal cancer", "mois mars bleu"],
        gt_seeds=["Mars bleu", "cancer colorectal", "depistage colon", "coloscopie"],
        gt_exclusions=[r"recette", r"peinture"],
    ),
    "DJ": Campaign(
        code="DJ", label="Dry January", color="#2a6fb0",
        window_start=(12, 26), window_end=(2, 6),
        gt_terms=["dry january", "mois sans alcool", "arreter de boire",
                  "janvier sans alcool", "arreter alcool"],
        mc_terms=["dry january", "consommation alcool", "mois alcool",
                  "boire alcool mois", "dry january janvier", "alcool mois janvier"],
        gt_seeds=["Dry January", "janvier sans alcool", "arreter de boire", "sevrage alcool"],
        gt_exclusions=[r"recette", r"cocktail", r"\bvin\b", r"\bbiere\b", r"whisky"],
    ),
}

ORDER = ["OR", "Mov", "MB", "DJ"]


def gt_query(c: Campaign) -> str:
    return " + ".join(c.gt_terms)


def mc_query(c: Campaign) -> str:
    return " OR ".join(f'"{t}"' for t in c.mc_terms)
