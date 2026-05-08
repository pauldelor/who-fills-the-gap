"""
fetch_fts.py — Download humanitarian aid flow records from OCHA FTS (Financial Tracking Service).

OCHA FTS tracks who gives money, who receives it, and for what purpose in humanitarian response.
This gives us the "front-line" view of aid: grants and contributions flowing into crises,
complementing the OECD DAC data which covers broader Official Development Assistance.

API: https://api.hpc.tools/v1/public/fts/flow
Coverage: 2018–2025 (pre- and post-US withdrawal)
Output:  data/raw/fts_flows_2018_2025.csv  (~218k rows)
Interim: data/interim/fts_YYYY.csv          (one file per year, for resumability)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.hpc.tools/v1/public/fts/flow"

# Years to pull: 2018–2025 gives four pre-cut years and one/two post-cut years.
# 2025 data will be partial (reporting is ongoing), which we'll flag in analysis.
YEARS = list(range(2018, 2026))

# How many flows to request per API call. 500 is the practical maximum;
# larger limits sometimes time-out on the server side.
PAGE_SIZE = 500

# Pause between requests so we don't hammer the OCHA server.
REQUEST_DELAY_SECONDS = 1.5

# Retry settings for transient connection errors (rate-limiting, resets).
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 10  # seconds; doubles on each retry

INTERIM_DIR = Path("data/interim")
RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "fts_flows_2018_2025.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_object(objects: list[dict], obj_type: str) -> Optional[str]:
    """
    Pull the `name` field from the first object matching obj_type.

    FTS flows carry nested 'sourceObjects' and 'destinationObjects' lists.
    Each item has a `type` key (e.g. "Organization", "Location", "GlobalCluster").
    We take the first match because most flows have only one of each type.
    """
    for obj in objects:
        if obj.get("type") == obj_type:
            return obj.get("name")
    return None


def _parse_flow(flow: dict) -> dict:
    """
    Flatten one raw FTS flow record into a tidy dictionary row.

    Key decisions:
    - Source org  = the entity writing the cheque (government, foundation, etc.)
    - Dest org    = the entity cashing it (UN agency, NGO, etc.)
    - Country     = the destination country the money is earmarked for
    - Cluster     = the humanitarian sector (Food Security, Health, WASH, etc.)
    - Usage year  = the year the money is intended to be *used* (≠ flow/budget year)
    """
    src_objs = flow.get("sourceObjects", [])
    dst_objs = flow.get("destinationObjects", [])

    # Source: find the first 'single' UsageYear as the reported year of the flow.
    usage_year = None
    for obj in src_objs + dst_objs:
        if obj.get("type") == "UsageYear" and obj.get("behavior") == "single":
            usage_year = obj.get("name")
            break

    return {
        "flow_id": flow.get("id"),
        "amount_usd": flow.get("amountUSD"),
        "budget_year": flow.get("budgetYear"),
        "usage_year": usage_year,
        "status": flow.get("status"),          # "commitment" | "paid"
        "flow_type": flow.get("flowType"),     # "Standard" | "Parked" | etc.
        "contribution_type": flow.get("contributionType"),  # "financial" | "in-kind"
        "method": flow.get("method"),
        "date": flow.get("date"),
        "source_org": _extract_object(src_objs, "Organization"),
        "source_country": _extract_object(src_objs, "Location"),
        "dest_org": _extract_object(dst_objs, "Organization"),
        "dest_country": _extract_object(dst_objs, "Location"),
        # GlobalCluster is the standardised humanitarian sector taxonomy.
        # Only present for ~30% of flows; the rest are unearmarked/pooled funds.
        "cluster": _extract_object(dst_objs, "GlobalCluster"),
    }


# ── Core fetch logic ─────────────────────────────────────────────────────────

def fetch_year(year: int, session: requests.Session) -> pd.DataFrame:
    """
    Download all FTS flows for a single year, paginating until exhausted.

    Returns a DataFrame with one row per flow.
    """
    interim_path = INTERIM_DIR / f"fts_{year}.csv"

    # Skip the download if we already have a cached file for this year.
    if interim_path.exists():
        print(f"  [cache] {year} — loading from {interim_path}")
        return pd.read_csv(interim_path)

    print(f"  [fetch] {year} — ", end="", flush=True)

    rows: list[dict] = []
    page = 1

    while True:
        params = {"year": year, "limit": PAGE_SIZE, "page": page}

        # Retry loop: handles transient connection resets from the OCHA server.
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(BASE_URL, params=params, timeout=60)
                resp.raise_for_status()
                break
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"\n    [retry {attempt+1}/{MAX_RETRIES-1}] {exc} — waiting {wait}s",
                      end=" ", flush=True)
                time.sleep(wait)

        body = resp.json()
        flows = body["data"]["flows"]
        total = body["meta"]["count"]

        rows.extend(_parse_flow(f) for f in flows)

        # Progress indicator: show pages fetched vs total expected
        total_pages = -(-total // PAGE_SIZE)  # ceiling division
        print(f"{page}/{total_pages}", end=" ", flush=True)

        # Stop when we've received fewer flows than the page size — last page.
        if len(flows) < PAGE_SIZE:
            break

        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"→ {len(rows):,} flows")

    df = pd.DataFrame(rows)
    df.to_csv(interim_path, index=False)
    print(f"    saved → {interim_path}")

    return df


def fetch_all(years: list[int] = YEARS) -> pd.DataFrame:
    """
    Fetch FTS flows for all specified years and write a combined CSV.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching OCHA FTS flows for {years[0]}–{years[-1]}\n")

    # Reuse one HTTP session so TCP connections are kept alive between pages.
    with requests.Session() as session:
        session.headers.update({"Accept": "application/json"})

        frames: list[pd.DataFrame] = []
        for year in years:
            df_year = fetch_year(year, session)
            # Tag each row with the query year so we can filter later even if
            # usage_year or budget_year is null.
            df_year["query_year"] = year
            frames.append(df_year)

    combined = pd.concat(frames, ignore_index=True)

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(combined):,} total rows → {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1_048_576:.1f} MB")

    return combined


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_all()

    print("\nQuick sanity check:")
    print(f"  Columns:  {list(df.columns)}")
    print(f"  Shape:    {df.shape}")
    print(f"  Nulls:\n{df.isnull().sum().to_string()}")
    print(f"\nTop source orgs:\n{df['source_org'].value_counts().head(10).to_string()}")
