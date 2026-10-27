"""Data collection: Google Trends (pytrends) and MediaCloud (mediacloud API).

Google Trends
    * ``gt_interest``      interest over time for one query and one time frame
    * ``snowball``         discovery of related queries over successive rounds
    * ``snowball_series``  GT signal of the cumulative query after each round
MediaCloud
    * ``mc_counts``        daily article counts (and collection totals)
    * ``mc_stories``       article metadata (URL, title, date), paginated
    * ``fetch_text``       full text of an article with newspaper4k
    * ``kgram_counts``     bigram / trigram frequencies used to build MC queries

pytrends is an unofficial client: Google rate-limits it aggressively (HTTP 429).
Every call therefore sleeps and retries with an increasing back-off.
"""
from __future__ import annotations

import datetime as dt
import random
import re
import time
from collections import Counter

import pandas as pd

from .config import GEO, MC_FRENCH_NATIONAL_COLLECTION, MEDIACLOUD_API_KEY, Campaign


# ---------------------------------------------------------------------------
# Google Trends
# ---------------------------------------------------------------------------
def gt_client():
    from pytrends.request import TrendReq
    return TrendReq(hl="fr-FR", tz=60, timeout=(10, 30))


def _gt_call(fn, retries: int = 5, base_wait: int = 60):
    """Run a pytrends call, waiting and retrying on rate-limit errors."""
    from pytrends.exceptions import ResponseError, TooManyRequestsError
    for attempt in range(retries):
        try:
            return fn()
        except (TooManyRequestsError, ResponseError) as e:
            wait = base_wait * (attempt + 1) + random.randint(0, 20)
            print(f"    Google Trends refused the request ({type(e).__name__}); "
                  f"retry {attempt + 1}/{retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Google Trends kept refusing the request; try again later.")


def gt_interest(client, query: str, timeframe: str, geo: str = GEO,
                pause: int = 20) -> pd.Series:
    """Interest over time (0-100) for one composite query ("a + b + c").

    The resolution depends on the length of ``timeframe``: daily below about
    nine months, weekly up to five years, monthly beyond.
    """
    def call():
        client.build_payload([query], timeframe=timeframe, geo=geo)
        return client.interest_over_time()
    df = _gt_call(call)
    time.sleep(pause + random.randint(0, 10))
    if df.empty:
        return pd.Series(dtype=float, name="value")
    s = df[query].astype(float)
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return s.rename("value")


def gt_related(client, term: str, timeframe: str, geo: str = GEO,
               pause: int = 45) -> dict[str, pd.DataFrame]:
    """Related queries ('top' and 'rising') of one term."""
    def call():
        client.build_payload([term], timeframe=timeframe, geo=geo)
        return client.related_queries()
    rq = _gt_call(call)
    time.sleep(pause + random.randint(0, 15))
    out = {}
    if rq and term in rq:
        for kind in ("top", "rising"):
            d = rq[term].get(kind)
            if d is not None and not d.empty:
                out[kind] = d
    return out


def _relevance(terms: list[str], anchors: list[str]) -> list[float]:
    """Max cosine similarity (character 3-5-grams, TF-IDF) to the seed terms.

    Used only to rank candidate terms for the manual review.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit(anchors + terms)
    sim = cosine_similarity(vec.transform(terms), vec.transform(anchors))
    return sim.max(axis=1).tolist()


def snowball(campaign: Campaign, timeframe: str, rounds: int = 3,
             max_new_per_round: int = 10, client=None) -> pd.DataFrame:
    """Iterative snowball sampling of Google Trends related queries.

    Round 0 is the list of seeds. In each round, the related queries of the
    terms found in the previous round are collected; excluded patterns are
    dropped. Only the ``max_new_per_round`` best new terms (by GT value) are
    explored in the next round, to keep the number of API calls bounded.

    Returns one row per term: term, round, found_via, kind, gt_value, relevance.
    The final query is chosen by hand from this table.
    """
    client = client or gt_client()
    excl = [re.compile(p, re.I) for p in campaign.gt_exclusions]
    seen = {s.lower() for s in campaign.gt_seeds}
    rows = [{"term": s, "round": 0, "found_via": "seed", "kind": "seed", "gt_value": 100.0}
            for s in campaign.gt_seeds]
    frontier = list(campaign.gt_seeds)
    for r in range(1, rounds + 1):
        print(f"  round {r}: exploring {len(frontier)} terms")
        found = []
        for term in frontier:
            for kind, d in gt_related(client, term, timeframe).items():
                for q, v in zip(d["query"], d["value"]):
                    q = str(q).strip()
                    if len(q) < 3 or q.lower() in seen or any(p.search(q) for p in excl):
                        continue
                    seen.add(q.lower())
                    found.append({"term": q, "round": r, "found_via": term, "kind": kind,
                                  "gt_value": float(v) if kind == "top" else float("nan")})
        rows += found
        top = sorted((f for f in found if f["kind"] == "top"),
                     key=lambda f: -f["gt_value"])[:max_new_per_round]
        frontier = [f["term"] for f in top]
        if not frontier:
            break
    out = pd.DataFrame(rows)
    out["relevance"] = _relevance(out["term"].tolist(), campaign.gt_seeds)
    return out.sort_values(["round", "gt_value"], ascending=[True, False]).reset_index(drop=True)


def snowball_series(campaign: Campaign, terms: pd.DataFrame, timeframe: str,
                    per_round: int = 2, client=None) -> pd.DataFrame:
    """GT signal of the cumulative query after each snowball round.

    The query of round r joins the campaign name and the ``per_round`` kept
    terms with the highest GT value from each round <= r. Use only the rows
    you validated (column ``keep`` == True if present).
    """
    client = client or gt_client()
    if "keep" in terms:
        terms = terms[terms["keep"].astype(bool)]
    query_terms = [campaign.gt_seeds[0]]
    out = []
    for r in sorted(t for t in terms["round"].unique() if t > 0):
        best = (terms[(terms["round"] == r) & (terms["kind"] == "top")]
                .nlargest(per_round, "gt_value")["term"].tolist())
        query_terms += [t for t in best if t not in query_terms]
        q = " + ".join(query_terms)
        s = gt_interest(client, q, timeframe)
        out.append(pd.DataFrame({"date": s.index, "value": s.values, "round": r, "query": q}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# MediaCloud
# ---------------------------------------------------------------------------
def mc_client():
    import mediacloud.api
    if not MEDIACLOUD_API_KEY:
        raise RuntimeError("Set the MEDIACLOUD_API_KEY environment variable before "
                           "starting Jupyter (export MEDIACLOUD_API_KEY=...).")
    return mediacloud.api.SearchApi(MEDIACLOUD_API_KEY)


def _mc_call(fn, retries: int = 4):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:                       # the client raises plain errors
            wait = 30 * (attempt + 1)
            print(f"    MediaCloud error ({e}); retry {attempt + 1}/{retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError("MediaCloud request failed repeatedly.")


def mc_counts(client, query: str, start: dt.date, end: dt.date,
              collection: int = MC_FRENCH_NATIONAL_COLLECTION) -> pd.DataFrame:
    """Daily number of matching articles ('count') and of all articles in the
    collection ('total_count'), from ``start`` to ``end`` inclusive.

    Requests are split by calendar year. Days without articles are filled with 0.
    """
    parts = []
    y = start.year
    while y <= end.year:
        a, b = max(start, dt.date(y, 1, 1)), min(end, dt.date(y, 12, 31))
        pts = _mc_call(lambda: client.story_count_over_time(query, a, b,
                                                            collection_ids=[collection]))
        parts.append(pd.DataFrame(pts))
        time.sleep(1)
        y += 1
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    days = pd.date_range(start, end, freq="D", name="date")
    if df.empty:
        return pd.DataFrame({"count": 0, "total_count": 0}, index=days)
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby("date")[["count", "total_count"]].sum()
    return df.reindex(days, fill_value=0)


def mc_stories(client, query: str, start: dt.date, end: dt.date,
               collection: int = MC_FRENCH_NATIONAL_COLLECTION,
               max_stories: int = 2000) -> pd.DataFrame:
    """Article metadata (id, title, url, media, date), following pagination."""
    rows, token = [], None
    while len(rows) < max_stories:
        page, token = _mc_call(lambda: client.story_list(
            query, start, end, collection_ids=[collection], pagination_token=token))
        rows += page
        if not token or not page:
            break
        time.sleep(1)
    return pd.DataFrame(rows[:max_stories])


def fetch_text(url: str, timeout: int = 15) -> str:
    """Full text of an article (empty string if download or parsing fails)."""
    from newspaper import Article
    try:
        art = Article(url, language="fr", request_timeout=timeout)
        art.download()
        art.parse()
        return art.text or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# k-grams
# ---------------------------------------------------------------------------
FRENCH_STOPWORDS = set("""
a à afin ai aie aient aies ait alors as au aucun aucune aupres auquel aura aurai auraient aurais
aurait auras aurez auriez aurions aurons auront aussi autre autres aux auxquelles auxquels avaient
avais avait avant avec avez aviez avions avoir avons ayant ayez ayons c ça car ce ceci cela celle
celles celui cependant certain certaine certains ces cet cette ceux chaque chez ci comme comment
d dans de des desquelles desquels dessous dessus deux devant doit donc dont du duquel durant elle
elles en encore entre es est et etaient etais etait etant ete etes etiez etions etre eu eue eues
eûmes eurent eus eusse eussent eusses eussiez eussions eut eux été étaient étais était étant
êtes être fait faire fois font furent fus fut ici il ils j je jusqu jusque l la laquelle le lequel
les lesquelles lesquels leur leurs lors lorsque lui m ma mais me même mêmes mes moi moins mon n ne
ni nos notre nous on ont ou où par parce pas peu peut plus pour pourquoi qu quand que quel quelle
quelles quels qui quoi s sa sans se selon ses si sien sienne siens soi soient sois soit sommes son
sont sous soyez soyons suis sur t ta te tes toi ton tous tout toute toutes tres très tu un une unes
uns vers voici voilà vos votre vous y
""".split())

# Lines typical of web boilerplate rather than editorial content (limitation (v)).
BOILERPLATE = re.compile(
    r"cookie|newsletter|abonn|inscri|connectez|publicit|partager|lire aussi|"
    r"tous droits|traceur|accepter|onglet|javascript", re.I)


def clean_text(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if l.strip() and not BOILERPLATE.search(l))


def kgram_counts(texts: list[str], ns=(2, 3), drop_boilerplate: bool = True,
                 min_count: int = 2) -> pd.DataFrame:
    """Frequency of word n-grams after removing French stop words.

    Stop words are removed before n-grams are formed, so "dépistage du cancer
    du sein" contributes "dépistage cancer sein".
    """
    counts = {n: Counter() for n in ns}
    for t in texts:
        if drop_boilerplate:
            t = clean_text(t)
        # n-grams are formed within a sentence, never across punctuation
        for sentence in re.split(r"[.!?;:\n()«»\"]+", t.lower().replace("’", "'")):
            words = [w.split("'")[-1] for w in re.findall(r"[a-zà-öø-ÿœæ'-]+", sentence)]
            words = [w for w in words if len(w) > 1 and w not in FRENCH_STOPWORDS]
            for n in ns:
                counts[n].update(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    rows = [{"n": n, "kgram": k, "count": c}
            for n in ns for k, c in counts[n].items() if c >= min_count]
    return pd.DataFrame(rows).sort_values(["n", "count"], ascending=[True, False],
                                          ignore_index=True)
