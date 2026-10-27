"""Medic'AM: monthly reimbursed drugs dispensed by community pharmacies (France).

Source: Assurance Maladie, "Medic'AM", published on data.gouv.fr
(dataset 536999d3a3a729239d20533a, all health insurance schemes). Monthly files
cover January 2015 onwards, one Excel file per half-year. Each row is a drug
presentation (CIP13 code) with its ATC classes and, for each month, the number
of reimbursed boxes, the reimbursement base and the amount reimbursed.

The parser below does not assume a fixed layout: it finds the header row by
looking for the CIP13 column, normalises column names (accents, case,
punctuation) and recognises monthly columns by their YYYY-MM suffix.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

DATAGOUV_DATASET = "536999d3a3a729239d20533a"
DATAGOUV_API = f"https://www.data.gouv.fr/api/1/datasets/{DATAGOUV_DATASET}/"

# ATC codes per campaign (paper slides, Annex 11). Regorafenib was L01XE21 before
# the 2022 ATC revision and L01EX05 since, so both codes are listed.
ATC = {
    "OR": {"L02BA01": "tamoxifen", "L02BG03": "anastrozole", "L02BG04": "letrozole",
           "L02BG06": "exemestane", "L02BA03": "fulvestrant"},
    "Mov": {"L02AE02": "leuprorelin", "L02AE03": "goserelin", "L02AE04": "triptorelin",
            "L02BB03": "bicalutamide", "L02BB04": "enzalutamide", "L02BX03": "abiraterone"},
    "MB": {"L01BC06": "capecitabine", "L01EX05": "regorafenib", "L01XE21": "regorafenib"},
    "DJ": {"N07BB01": "disulfiram", "N07BB03": "acamprosate", "N07BB04": "naltrexone",
           "N07BB05": "nalmefene"},
}

MEASURES = {"boites": "boxes", "montant": "amount_reimbursed", "base": "reimbursement_base"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def list_resources() -> pd.DataFrame:
    """Files of the data.gouv.fr dataset (title, format, url, last_modified)."""
    import requests
    r = requests.get(DATAGOUV_API, timeout=60)
    r.raise_for_status()
    res = pd.DataFrame(r.json()["resources"])
    keep = [c for c in ("title", "format", "url", "last_modified", "filesize") if c in res]
    return res[keep].sort_values("title", ignore_index=True)


def download(resources: pd.DataFrame, dest: Path, overwrite: bool = False) -> list[Path]:
    """Download the selected resources into ``dest`` (one file per row)."""
    import requests
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for title, url, fmt in zip(resources["title"], resources["url"], resources["format"]):
        name = re.sub(r"[^\w\-. ]", "_", str(title)).strip()
        ext = "." + str(fmt).lower().lstrip(".")
        path = dest / (name if name.lower().endswith(ext) else name + ext)
        if path.exists() and not overwrite:
            paths.append(path)
            continue
        print("  downloading", path.name)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def norm(s) -> str:
    """'Nombre de boîtes remboursées 2017-07' -> 'nombre_de_boites_remboursees_2017_07'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _header_row(raw: pd.DataFrame, max_rows: int = 40) -> int | None:
    for i in range(min(max_rows, len(raw))):
        if any(norm(v).startswith("cip") for v in raw.iloc[i].tolist()):
            return i
    return None


def _pick(cols: list[str], *patterns: str) -> str | None:
    for p in patterns:
        for c in cols:
            if re.fullmatch(p, c):
                return c
    return None


def inspect_file(path: Path) -> pd.DataFrame:
    """One row per sheet: header row found, number of monthly columns, months covered."""
    rows = []
    for sheet, raw in pd.read_excel(path, sheet_name=None, header=None, dtype=str).items():
        h = _header_row(raw)
        cols = [norm(c) for c in raw.iloc[h]] if h is not None else []
        months = sorted({m.group(1) + "-" + m.group(2) for c in cols
                         if (m := re.search(r"(\d{4})_(\d{2})$", c))})
        rows.append({"file": path.name, "sheet": sheet, "header_row": h,
                     "rows": len(raw) - (h or 0) - 1, "monthly_columns": len(months),
                     "first_month": months[0] if months else None,
                     "last_month": months[-1] if months else None,
                     "columns": ", ".join(c for c in cols if not re.search(r"\d{4}_\d{2}$", c))})
    return pd.DataFrame(rows)


def read_sheet(path: Path, sheet) -> pd.DataFrame:
    """Long table of one sheet: cip13, name, atc, date, measure, value."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    h = _header_row(raw)
    if h is None:
        raise ValueError(f"No CIP13 header found in {path.name}, sheet {sheet!r}")
    df = raw.iloc[h + 1:].copy()
    df.columns = [norm(c) for c in raw.iloc[h]]
    cols = list(df.columns)
    c_cip = _pick(cols, r"cip_?13", r"cip.*")
    c_atc = _pick(cols, r"code_atc", r"code_atc_?5", r"atc_?5", r"code_6", r"atc")
    c_name = _pick(cols, r"nom_court", r"libelle.*", r"nom.*")
    if c_cip is None or c_atc is None:
        raise ValueError(f"CIP13 or ATC column not found in {path.name}: {cols}")
    # drop summary rows ("Total", "Homéopathie"...): a CIP13 is made of digits
    df = df[df[c_cip].astype(str).str.fullmatch(r"\d{6,13}(\.0)?")].copy()
    df["cip13"] = df[c_cip].str.replace(r"\.0$", "", regex=True).str.zfill(13)
    df["atc"] = df[c_atc].str.strip().str.upper()
    df["name"] = df[c_name] if c_name else ""
    monthly = {}
    for c in cols:
        m = re.search(r"(\d{4})_(\d{2})$", c)
        if not m:
            continue
        measure = next((v for k, v in MEASURES.items() if k in c), None)
        if measure:
            monthly[c] = (measure, f"{m.group(1)}-{m.group(2)}-01")
    long = df.melt(id_vars=["cip13", "name", "atc"], value_vars=list(monthly),
                   var_name="column", value_name="value")
    long["measure"] = long["column"].map(lambda c: monthly[c][0])
    long["date"] = pd.to_datetime(long["column"].map(lambda c: monthly[c][1]))
    long["value"] = pd.to_numeric(long["value"].str.replace(r"[\s ]", "", regex=True)
                                  .str.replace(",", "."), errors="coerce")
    long["file"], long["sheet"] = path.name, sheet
    return long.drop(columns="column")


def parse_folder(folder: Path, atc_codes: list[str], sheet=1) -> pd.DataFrame:
    """Parse every .xls / .xlsx file of ``folder`` and keep the given ATC codes.

    sheet : sheet to read in each file (index or name). The public R parser of
            Medic'AM files reads the second sheet (index 1); check with
            ``inspect_file`` which sheet holds the totals you want.
    If the same month appears in several files, the most recent file (by name
    order, then modification time) is kept.
    """
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".xls", ".xlsx")],
                   key=lambda p: (p.stat().st_mtime, p.name))
    if not files:
        raise FileNotFoundError(f"No .xls/.xlsx file in {folder}")
    parts = []
    for p in files:
        d = read_sheet(p, sheet)
        parts.append(d[d["atc"].isin(atc_codes)])
        print(f"  {p.name}: {d['date'].min():%Y-%m} → {d['date'].max():%Y-%m}, "
              f"{d['cip13'].nunique()} presentations")
    out = pd.concat(parts, ignore_index=True)
    n = len(out)
    out = out.drop_duplicates(["cip13", "date", "measure"], keep="last")
    if len(out) < n:
        print(f"  {n - len(out)} duplicated (presentation, month) values replaced by the most recent file")
    return out.sort_values(["atc", "cip13", "measure", "date"], ignore_index=True)


def monthly_by_atc(long: pd.DataFrame, measure: str = "boxes") -> pd.DataFrame:
    """Monthly totals per ATC code (wide: one column per ATC code)."""
    d = long[long["measure"] == measure]
    return d.pivot_table(index="date", columns="atc", values="value", aggfunc="sum").sort_index()


def monthly_by_campaign(long: pd.DataFrame, measure: str = "boxes") -> pd.DataFrame:
    by_atc = monthly_by_atc(long, measure)
    return pd.DataFrame({code: by_atc[[a for a in codes if a in by_atc]].sum(axis=1, min_count=1)
                         for code, codes in ATC.items()})


def stl_shares(series: pd.Series, period: int = 12) -> dict:
    """STL decomposition y = T + S + R and the share of each component.

    Shares are var(component) / [var(T) + var(S) + var(R)] (they sum to 100%).
    cov_share = [var(y) - var(T) - var(S) - var(R)] / var(y): the part of the
    variance due to covariances between components (0 if they are uncorrelated).
    """
    from statsmodels.tsa.seasonal import STL
    y = series.dropna().asfreq("MS")
    res = STL(y, period=period, robust=True).fit()
    v = {"trend": res.trend.var(), "seasonal": res.seasonal.var(), "remainder": res.resid.var()}
    tot = sum(v.values())
    out = {k: 100 * x / tot for k, x in v.items()}
    out["cov_share"] = 100 * (y.var() - tot) / y.var()
    return out, res
