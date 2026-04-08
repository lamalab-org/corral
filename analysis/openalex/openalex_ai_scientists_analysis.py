"""OpenAlex-based analysis of AI-scientist / agentic-science papers in chemistry
and materials science.

This script is intentionally narrower than a general AI-in-science analysis:
it targets papers about AI scientists, agentic science, LLM agents,
autonomous scientific discovery, and self-driving laboratories.

Outputs:
- works_all_hits.csv         deduplicated retrieved papers
- works_strict.csv           strict bucket only
- works_expanded.csv         strict + expanded bucket
- yearly_counts.csv          yearly paper counts and shares
- yearly_authors.csv         yearly unique-author counts
- summary.json               metadata and query definitions used

Example:
    python openalex_ai_scientists_analysis.py \
        --api-key YOUR_OPENALEX_KEY \
        --start-year 2018 \
        --end-year 2026 \
        --outdir results_ai_scientists
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://api.openalex.org/works"
MAX_PER_PAGE = 100
REQUEST_TIMEOUT = 60
RETRY_STATUSES = {429, 500, 502, 503, 504}
DEFAULT_TYPES = ("article", "review", "preprint")
DEFAULT_FIELD_IDS = ("16", "25")  # 16 = Chemistry, 25 = Materials Science

CHEM_MAT_TERMS = [
    r"chemistry",
    r"chemical",
    r"molecular",
    r"molecule(?:s)?",
    r"materials?",
    r"materials science",
    r"synthesis",
    r"catalyst(?:s|ic)?",
    r"catalysis",
    r"polymer(?:s)?",
    r"battery",
    r"semiconductor(?:s)?",
    r"crystal(?:s|line)?",
    r"alloy(?:s)?",
    r"electrolyte(?:s)?",
    r"reaction(?:s)?",
]
CHEM_MAT_RE = re.compile(r"\b(?:" + "|".join(CHEM_MAT_TERMS) + r")\b", re.IGNORECASE)

STRICT_PATTERNS = {
    "ai_scientist": [
        r"\bai scientist(?:s)?\b",
        r"\bartificial intelligence scientist(?:s)?\b",
    ],
    "agentic_science": [
        r"\bagentic science\b",
    ],
    "llm_agents": [
        r"\bllm agent(?:s)?\b",
        r"\blarge language model agent(?:s)?\b",
        r"\blanguage model agent(?:s)?\b",
    ],
    "scientific_agents": [
        r"\bscientific agent(?:s)?\b",
        r"\bresearch agent(?:s)?\b",
        r"\bdiscovery agent(?:s)?\b",
        r"\bai agent(?:s)?\b",
        r"\bautonomous agent(?:s)?\b",
        r"\bmulti-agent\b",
        r"\bmulti agent\b",
    ],
    "self_driving_labs": [
        r"\bself[- ]driving lab(?:orator(?:y|ies))?\b",
        r"\bautonomous experiment(?:ation|s)?\b",
        r"\bclosed[- ]loop discovery\b",
        r"\bclosed[- ]loop optimization\b",
    ],
}

EXPANDED_PATTERNS = {
    "autonomous_discovery": [
        r"\bautonomous scientific discovery\b",
        r"\bautonomous discovery\b",
        r"\bscientific discovery\b",
    ],
    "agentic_language": [
        r"\bagentic\b",
        r"\btool[- ]using agent(?:s)?\b",
        r"\btool use agent(?:s)?\b",
        r"\bagent[- ]based scientific discovery\b",
    ],
}

STRICT_QUERIES = [
    (
        "strict_ai_scientist",
        '"ai scientist" AND '
        "(chemistry OR chemical OR molecular OR molecule* OR materials OR material* "
        "OR synthesis OR catalyst OR catalysis OR battery OR polymer OR semiconductor OR crystal OR alloy)",
    ),
    (
        "strict_agentic_science",
        '"agentic science" AND '
        "(chemistry OR chemical OR molecular OR materials OR material* OR synthesis OR catalyst "
        "OR battery OR polymer OR semiconductor OR crystal OR alloy)",
    ),
    (
        "strict_llm_agents",
        '(("llm agent" OR "large language model agent" OR "language model agent" '
        'OR "ai agent" OR "autonomous agent" OR "multi-agent") '
        "AND (science OR scientific OR discovery OR experiment* OR planning) "
        "AND (chemistry OR chemical OR molecular OR materials OR material* OR synthesis "
        "OR catalyst OR battery OR polymer OR semiconductor OR crystal OR alloy))",
    ),
    (
        "strict_self_driving_labs",
        '(("self-driving lab" OR "self-driving laboratory" OR "autonomous experimentation" '
        'OR "closed-loop discovery" OR "closed-loop optimization") '
        "AND (chemistry OR chemical OR molecular OR materials OR material* OR synthesis "
        "OR catalyst OR battery OR polymer OR semiconductor OR crystal OR alloy))",
    ),
]

EXPANDED_QUERIES = [
    (
        "expanded_autonomous_scientific_discovery",
        '(("autonomous scientific discovery" OR "scientific discovery") '
        'AND (agent OR agents OR autonomous OR "large language model" OR llm) '
        "AND (chemistry OR chemical OR molecular OR materials OR material* OR synthesis "
        "OR catalyst OR battery OR polymer OR semiconductor OR crystal OR alloy))",
    ),
    (
        "expanded_research_discovery_agents",
        '(("research agent" OR "scientific agent" OR "discovery agent" OR "tool-using agent") '
        "AND (chemistry OR chemical OR molecular OR materials OR material* OR synthesis "
        "OR catalyst OR battery OR polymer OR semiconductor OR crystal OR alloy))",
    ),
    (
        "expanded_llm_multi_agent_science",
        '(("foundation model" OR "large language model" OR llm OR transformer) '
        'AND (agent OR agents OR autonomous OR "multi-agent") '
        "AND (science OR scientific OR discovery OR experiment* OR planning) "
        "AND (chemistry OR chemical OR molecular OR materials OR material* OR synthesis "
        "OR catalyst OR battery OR polymer OR semiconductor OR crystal OR alloy))",
    ),
]


def compile_patterns(
    pattern_map: dict[str, Sequence[str]],
) -> dict[str, list[re.Pattern[str]]]:
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for key, patterns in pattern_map.items():
        compiled[key] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return compiled


STRICT_RE = compile_patterns(STRICT_PATTERNS)
EXPANDED_RE = compile_patterns(EXPANDED_PATTERNS)


@dataclass
class QuerySpec:
    bucket: str  # strict or expanded
    label: str
    query: str


ALL_QUERY_SPECS: list[QuerySpec] = [
    QuerySpec(bucket="strict", label=label, query=query)
    for label, query in STRICT_QUERIES
] + [
    QuerySpec(bucket="expanded", label=label, query=query)
    for label, query in EXPANDED_QUERIES
]


class OpenAlexClient:
    def __init__(self, api_key: str, sleep_seconds: float = 0.15, max_retries: int = 6):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "openalex-ai-scientists-analysis/1.0",
                "Accept": "application/json",
            }
        )

    def get(self, params: dict[str, str]) -> dict:
        params = dict(params)
        params["api_key"] = self.api_key
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    BASE_URL, params=params, timeout=REQUEST_TIMEOUT
                )
                if response.status_code in RETRY_STATUSES:
                    wait = min(2**attempt, 30)
                    logger.warning(
                        "OpenAlex returned {}; retrying in {}s",
                        response.status_code,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                payload = response.json()
                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)
                return payload
            except requests.RequestException as exc:
                last_error = exc
                wait = min(2**attempt, 30)
                logger.warning("Request failed ({}); retrying in {}s", exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenAlex request failed after retries: {last_error}")

    def iterate_works(self, search_query: str, filter_str: str) -> Iterable[dict]:
        cursor = "*"
        while cursor:
            params = {
                "search": search_query,
                "filter": filter_str,
                "per_page": str(MAX_PER_PAGE),
                "cursor": cursor,
            }
            payload = self.get(params)
            yield from payload.get("results", [])
            cursor = payload.get("meta", {}).get("next_cursor")
            if not payload.get("results"):
                break

    def group_by_publication_year(self, filter_str: str) -> pd.DataFrame:
        params = {
            "filter": filter_str,
            "group_by": "publication_year",
            "per_page": "200",
        }
        payload = self.get(params)
        rows = []
        for group in payload.get("group_by", []):
            try:
                year = int(group["key"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {"publication_year": year, "total_field_papers": int(group["count"])}
            )
        return pd.DataFrame(rows)


def reconstruct_abstract(inverted_index: dict[str, Sequence[int]] | None) -> str:
    if not inverted_index:
        return ""
    max_pos = -1
    for positions in inverted_index.values():
        if positions:
            max_pos = max(max_pos, *positions)
    if max_pos < 0:
        return ""
    tokens = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(tokens):
                tokens[pos] = word
    return " ".join(tok for tok in tokens if tok).strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def short_openalex_id(raw_id: str | None) -> str:
    if not raw_id:
        return ""
    return raw_id.rstrip("/").split("/")[-1]


def extract_topics(work: dict) -> list[str]:
    return [
        t.get("display_name", "")
        for t in work.get("topics", [])
        if t.get("display_name")
    ]


def extract_topic_fields(work: dict) -> list[str]:
    out = set()
    primary_field = ((work.get("primary_topic") or {}).get("field") or {}).get(
        "display_name"
    )
    if primary_field:
        out.add(primary_field)
    for topic in work.get("topics", []):
        field_name = (topic.get("field") or {}).get("display_name")
        if field_name:
            out.add(field_name)
    return sorted(out)


def extract_keywords(work: dict) -> list[str]:
    return [
        k.get("display_name", "")
        for k in work.get("keywords", [])
        if k.get("display_name")
    ]


def extract_authors(work: dict) -> tuple[list[str], list[str], list[str]]:
    names, author_ids, countries = [], [], []
    for authorship in work.get("authorships", []):
        author = authorship.get("author") or {}
        if author.get("display_name"):
            names.append(author["display_name"])
        if author.get("id"):
            author_ids.append(short_openalex_id(author["id"]))
        countries.extend(c for c in (authorship.get("countries", []) or []) if c)
    # preserve order while deduplicating
    names = list(dict.fromkeys(names))
    author_ids = list(dict.fromkeys(author_ids))
    countries = list(dict.fromkeys(countries))
    return names, author_ids, countries


def extract_source_name(work: dict) -> str:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return source.get("display_name", "") or ""


def combined_text(work: dict) -> str:
    title = work.get("display_name") or work.get("title") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    keywords = " ; ".join(extract_keywords(work))
    topics = " ; ".join(extract_topics(work))
    parts = [title, abstract, keywords, topics]
    return normalize_space(" ".join(part for part in parts if part))


def match_pattern_groups(
    text: str, compiled_map: dict[str, list[re.Pattern[str]]]
) -> list[str]:
    matched = []
    for label, patterns in compiled_map.items():
        if any(p.search(text) for p in patterns):
            matched.append(label)
    return matched


def is_chem_or_materials(work: dict, text: str) -> bool:
    fields = set(extract_topic_fields(work))
    if "Chemistry" in fields or "Materials Science" in fields:
        return True
    return bool(CHEM_MAT_RE.search(text))


def classify_tier(
    text: str,
    originating_buckets: Sequence[str],
) -> tuple[str | None, list[str]]:
    strict_hits = match_pattern_groups(text, STRICT_RE)
    expanded_hits = match_pattern_groups(text, EXPANDED_RE)
    evidence = strict_hits + expanded_hits

    if "strict" in originating_buckets and (strict_hits or expanded_hits):
        return "strict", evidence
    if strict_hits:
        return "strict", evidence
    if expanded_hits or "expanded" in originating_buckets:
        return "expanded", evidence
    return None, evidence


def build_filter(
    start_year: int,
    end_year: int,
    field_ids: Sequence[str] | None,
    types: Sequence[str],
) -> str:
    parts = [f"publication_year:{start_year}-{end_year}"]
    if types:
        parts.append("type:" + "|".join(types))
    if field_ids:
        parts.append("primary_topic.field.id:" + "|".join(field_ids))
    return ",".join(parts)


def flatten_work_record(
    work: dict,
    tier: str,
    evidence: Sequence[str],
    matched_query_labels: Sequence[str],
    matched_buckets: Sequence[str],
) -> dict:
    text = combined_text(work)
    author_names, author_ids, countries = extract_authors(work)
    primary_topic = work.get("primary_topic") or {}
    primary_field = ((primary_topic.get("field") or {}).get("display_name")) or ""
    primary_subfield = ((primary_topic.get("subfield") or {}).get("display_name")) or ""
    return {
        "openalex_id": short_openalex_id(work.get("id")),
        "openalex_url": work.get("id", ""),
        "doi": work.get("doi", ""),
        "title": work.get("display_name") or work.get("title") or "",
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date", ""),
        "type": work.get("type", ""),
        "cited_by_count": work.get("cited_by_count", 0),
        "source": extract_source_name(work),
        "primary_field": primary_field,
        "primary_subfield": primary_subfield,
        "topic_fields": " | ".join(extract_topic_fields(work)),
        "topics": " | ".join(extract_topics(work)),
        "keywords": " | ".join(extract_keywords(work)),
        "authors_count": len(author_ids),
        "author_ids": " | ".join(author_ids),
        "authors": " | ".join(author_names),
        "countries": " | ".join(countries),
        "tier": tier,
        "matched_query_buckets": " | ".join(sorted(set(matched_buckets))),
        "matched_queries": " | ".join(sorted(set(matched_query_labels))),
        "match_evidence": " | ".join(sorted(set(evidence))),
        "text": text,
    }


def fetch_and_classify(
    client: OpenAlexClient,
    query_specs: Sequence[QuerySpec],
    filter_str: str,
) -> pd.DataFrame:
    aggregated: dict[str, dict] = {}

    for idx, spec in enumerate(query_specs, start=1):
        logger.info("[{}/{}] Searching {}", idx, len(query_specs), spec.label)
        n_seen = 0
        for work in client.iterate_works(spec.query, filter_str):
            n_seen += 1
            work_id = short_openalex_id(work.get("id"))
            if not work_id:
                continue
            entry = aggregated.setdefault(
                work_id,
                {
                    "work": work,
                    "matched_queries": set(),
                    "matched_buckets": set(),
                },
            )
            entry["matched_queries"].add(spec.label)
            entry["matched_buckets"].add(spec.bucket)
            # retain the most recent copy if API varies slightly across searches
            entry["work"] = work
        logger.debug("Raw hits fetched: {}", n_seen)

    rows = []
    for entry in aggregated.values():
        work = entry["work"]
        text = combined_text(work)
        if not text:
            continue
        if not is_chem_or_materials(work, text):
            continue
        tier, evidence = classify_tier(text, sorted(entry["matched_buckets"]))
        if tier is None:
            continue
        rows.append(
            flatten_work_record(
                work=work,
                tier=tier,
                evidence=evidence,
                matched_query_labels=sorted(entry["matched_queries"]),
                matched_buckets=sorted(entry["matched_buckets"]),
            )
        )

    if not rows:
        return pd.DataFrame()

    papers_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["openalex_id"])
        .sort_values(["publication_year", "cited_by_count"], ascending=[True, False])
    )
    return papers_df.reset_index(drop=True)


def compute_yearly_counts(
    df: pd.DataFrame, totals_by_year: pd.DataFrame
) -> pd.DataFrame:
    all_years = pd.DataFrame(
        {
            "publication_year": sorted(
                df["publication_year"].dropna().astype(int).unique()
            )
        }
    )

    strict_counts = (
        df[df["tier"] == "strict"]
        .groupby("publication_year", as_index=False)
        .size()
        .rename(columns={"size": "strict_papers"})
    )
    expanded_counts = (
        df.groupby("publication_year", as_index=False)
        .size()
        .rename(columns={"size": "expanded_papers"})
    )

    out = all_years.merge(strict_counts, on="publication_year", how="left")
    out = out.merge(expanded_counts, on="publication_year", how="left")
    out = out.merge(totals_by_year, on="publication_year", how="left")
    out[["strict_papers", "expanded_papers", "total_field_papers"]] = out[
        ["strict_papers", "expanded_papers", "total_field_papers"]
    ].fillna(0)
    out["strict_share_of_field"] = out.apply(
        lambda r: (r["strict_papers"] / r["total_field_papers"])
        if r["total_field_papers"]
        else 0.0,
        axis=1,
    )
    out["expanded_share_of_field"] = out.apply(
        lambda r: (r["expanded_papers"] / r["total_field_papers"])
        if r["total_field_papers"]
        else 0.0,
        axis=1,
    )
    return out.sort_values("publication_year").reset_index(drop=True)


def compute_yearly_authors(df: pd.DataFrame) -> pd.DataFrame:
    author_rows = []
    for _, row in df.iterrows():
        year = row["publication_year"]
        tier = row["tier"]
        ids = [x.strip() for x in str(row["author_ids"]).split("|") if x.strip()]
        author_rows.extend(
            {"publication_year": year, "author_id": author_id, "tier": tier}
            for author_id in ids
        )
    if not author_rows:
        return pd.DataFrame(
            columns=[
                "publication_year",
                "strict_unique_authors",
                "expanded_unique_authors",
            ]
        )

    adf = pd.DataFrame(author_rows).drop_duplicates()
    strict_authors = (
        adf[adf["tier"] == "strict"]
        .groupby("publication_year")["author_id"]
        .nunique()
        .reset_index(name="strict_unique_authors")
    )
    expanded_authors = (
        adf.groupby("publication_year")["author_id"]
        .nunique()
        .reset_index(name="expanded_unique_authors")
    )
    out = strict_authors.merge(
        expanded_authors, on="publication_year", how="outer"
    ).fillna(0)
    out[["strict_unique_authors", "expanded_unique_authors"]] = out[
        ["strict_unique_authors", "expanded_unique_authors"]
    ].astype(int)
    return out.sort_values("publication_year").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENALEX_API_KEY"),
        help="OpenAlex API key. Can also be set via OPENALEX_API_KEY.",
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--outdir", type=Path, default=Path("openalex_ai_scientists_output")
    )
    parser.add_argument(
        "--field-ids",
        default=",".join(DEFAULT_FIELD_IDS),
        help="Comma-separated OpenAlex field IDs used for primary_topic filtering. Default: 16,25 (Chemistry, Materials Science). Use '' to disable field filtering.",
    )
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="Comma-separated work types. Default: article,review,preprint",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Delay between successful API requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        logger.error("Provide --api-key or set OPENALEX_API_KEY")
        return 2
    if args.start_year > args.end_year:
        logger.error("--start-year must be <= --end-year")
        return 2

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    field_ids = [x.strip() for x in str(args.field_ids).split(",") if x.strip()]
    types = [x.strip() for x in str(args.types).split(",") if x.strip()]
    filter_str = build_filter(
        args.start_year, args.end_year, field_ids if field_ids else None, types
    )

    logger.info("Using filter: {}", filter_str)
    client = OpenAlexClient(api_key=args.api_key, sleep_seconds=args.sleep_seconds)

    works_df = fetch_and_classify(
        client=client, query_specs=ALL_QUERY_SPECS, filter_str=filter_str
    )
    if works_df.empty:
        logger.warning(
            "No papers matched after classification. Consider loosening the year range or keyword set."
        )
        summary = {
            "status": "no_matches",
            "filter": filter_str,
            "queries": [spec.__dict__ for spec in ALL_QUERY_SPECS],
        }
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return 0

    totals_by_year = client.group_by_publication_year(filter_str=filter_str)
    yearly_counts = compute_yearly_counts(works_df, totals_by_year)
    yearly_authors = compute_yearly_authors(works_df)

    strict_df = works_df[works_df["tier"] == "strict"].copy()
    expanded_df = works_df.copy()

    works_df.to_csv(outdir / "works_all_hits.csv", index=False)
    strict_df.to_csv(outdir / "works_strict.csv", index=False)
    expanded_df.to_csv(outdir / "works_expanded.csv", index=False)
    yearly_counts.to_csv(outdir / "yearly_counts.csv", index=False)
    yearly_authors.to_csv(outdir / "yearly_authors.csv", index=False)

    summary = {
        "status": "ok",
        "filter": filter_str,
        "n_total_retrieved": len(works_df),
        "n_strict": len(strict_df),
        "n_expanded": len(expanded_df),
        "year_range": [args.start_year, args.end_year],
        "field_ids": field_ids,
        "types": types,
        "queries": [spec.__dict__ for spec in ALL_QUERY_SPECS],
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info("Saved outputs to: {}", outdir)
    logger.info("  works_all_hits.csv  ({} rows)", len(works_df))
    logger.info("  works_strict.csv    ({} rows)", len(strict_df))
    logger.info("  works_expanded.csv  ({} rows)", len(expanded_df))
    logger.info("  yearly_counts.csv")
    logger.info("  yearly_authors.csv")
    logger.info("  summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
