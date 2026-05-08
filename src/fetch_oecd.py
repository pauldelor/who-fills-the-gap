"""
fetch_oecd.py

Downloads OECD DAC CRS (Creditor Reporting System) data for 2018-2024.
The CRS tracks individual aid projects/flows from DAC donor countries.

We filter to:
  - ODA flows only (Official Development Assistance, FlowCode = 11)
  - Key columns only (see KEEP_COLUMNS below)

Output: data/raw/oecd_dac_2018_2024.csv

HOW IT WORKS:
  1. Direct bulk-file URLs are embedded in OECD's SDMX dataflow metadata.
     Each year's data is a ZIP (~50-60 MB) containing a delimited text file.
  2. We download each year's ZIP, filter it in memory, and save a small
     interim CSV per year to data/raw/ (so re-runs skip already-done years).
  3. All years are concatenated and saved as data/raw/oecd_dac_2018_2024.csv.

DATA SOURCE:
  OECD DAC CRS — https://sdmx.oecd.org/public/rest/dataflow/OECD.DCD.FSD/DSD_CRS@DF_CRS
  (Direct download URLs discovered from dataflow EXT_RESOURCE annotations, April 2026.)
"""

# ============================================================
# IMPORTS
# ============================================================
import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE  = RAW_DIR / "oecd_dac_2018_2024.csv"

# Years we want to cover
YEARS = list(range(2018, 2025))   # 2018 through 2024 inclusive

# ODA = Official Development Assistance; FlowCode 11 in OECD DAC conventions
ODA_FLOW_CODE = 11

# The 7 columns we need from the raw data (raw files have 50+ columns)
KEEP_COLUMNS = [
    "Year",
    "DonorName",
    "RecipientName",
    "SectorName",
    "USD_Disbursement",
    "USD_Commitment",
    "FlowName",
]

# ---- Direct download URLs, one per year ----
# Source: EXT_RESOURCE annotations in the OECD SDMX dataflow metadata
# Discovered programmatically from:
#   https://sdmx.oecd.org/public/rest/dataflow/OECD.DCD.FSD/DSD_CRS@DF_CRS
# Version tag in URL filename: v20260408 (last updated April 8 2026)
# Each URL is a ~50-60 MB ZIP containing a delimited text file.
YEAR_URLS: Dict[int, str] = {
    2024: "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=fad5450f-e2ef-4f2b-ad7a-f9d70c8f6124",
    2023: "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=2c5b7733-44c8-4c4c-a59b-1f7c2d75df8c",
    2022: "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=0f9670bd-5900-4e07-b49a-384890334534",
    2021: "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=d85be660-7811-45c1-b807-810b42055259",
    2020: "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=ac3a59f6-5acb-4747-9ef3-3c00e2ab3eb5",
    2019: "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=348f223c-7d11-43ad-ab95-b7c0ab537fdd",
    2018: "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=ed94e89a-e81e-4718-ac69-2a51f5429657",
}

# Bytes to read before showing progress update (~10 MB chunks)
CHUNK_SIZE = 10 * 1024 * 1024

# Seconds before giving up on a single HTTP request
REQUEST_TIMEOUT = 300


# ============================================================
# UTILITY
# ============================================================

def ensure_dirs():
    """Create data/raw/ if it does not already exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _interim_path(year: int) -> Path:
    """Return the path for a cached per-year filtered CSV."""
    return RAW_DIR / f"oecd_crs_{year}_oda.csv"


def output_already_exists() -> bool:
    """
    Return True if the final combined CSV exists and has content.
    This prevents re-running the full download when nothing has changed.
    """
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        mb = OUTPUT_FILE.stat().st_size / 1_048_576
        print(f"[SKIP] Output already exists: {OUTPUT_FILE.name} ({mb:.1f} MB)")
        print("       Delete it to force a fresh download.\n")
        return True
    return False


# ============================================================
# DOWNLOAD — stream with progress
# ============================================================

def download_year_zip(year: int) -> Optional[bytes]:
    """
    Download one year's bulk ZIP file from OECD.Stat and return
    the raw bytes. Shows download progress as it streams.

    The ZIP files are 50-60 MB each — large enough to warrant streaming
    rather than loading the full response body at once.

    Args:
        year: The calendar year to download (e.g. 2023)

    Returns:
        ZIP file bytes if successful, None if the download failed.
    """
    url = YEAR_URLS[year]
    print(f"\n[{year}] Downloading from OECD... ", flush=True)

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()

        # Stream the file in chunks and show progress
        total_bytes  = int(response.headers.get("Content-Length", 0))
        chunks       = []
        downloaded   = 0

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                # Show progress as a simple percentage
                if total_bytes:
                    pct = downloaded / total_bytes * 100
                    print(f"\r[{year}] {downloaded / 1_048_576:.0f} / {total_bytes / 1_048_576:.0f} MB  ({pct:.0f}%)  ", end="", flush=True)

        print()   # newline after progress line
        return b"".join(chunks)

    except requests.Timeout:
        print(f"\n[{year}] TIMED OUT after {REQUEST_TIMEOUT}s")
        return None
    except Exception as exc:
        print(f"\n[{year}] DOWNLOAD ERROR: {exc}")
        return None


# ============================================================
# EXTRACT — read the text file inside the ZIP
# ============================================================

def extract_csv_from_zip(zip_bytes: bytes, year: int) -> Optional[pd.DataFrame]:
    """
    Open a ZIP file from memory and read the data file inside it.

    OECD bulk ZIPs contain one delimited text file (extension .txt or .csv).
    The file uses comma-separated values in the 'dotStat' format —
    similar to a standard CSV with a header row.

    Args:
        zip_bytes: Raw bytes of the downloaded ZIP file
        year:      Used only for log messages

    Returns:
        Raw DataFrame if successful, None if extraction failed.
    """
    print(f"[{year}] Extracting from ZIP...", flush=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Find all files inside the ZIP (there is usually just one)
            all_names = zf.namelist()
            print(f"[{year}] Files inside ZIP: {all_names}")

            # Accept .txt and .csv extensions
            data_files = [
                n for n in all_names
                if n.lower().endswith(".txt") or n.lower().endswith(".csv")
            ]

            if not data_files:
                print(f"[{year}] ERROR: No data file found inside ZIP")
                return None

            # Read the first data file found
            target_file = data_files[0]
            print(f"[{year}] Reading '{target_file}'...", flush=True)

            with zf.open(target_file) as f:
                # OECD dotStat bulk files use PIPE (|) as delimiter.
                # The header row has quoted column names; data rows are unquoted.
                # on_bad_lines='warn' skips malformed rows instead of crashing.
                df = pd.read_csv(
                    f,
                    sep="|",
                    encoding="utf-8",
                    low_memory=False,
                    on_bad_lines="warn",
                )

        print(f"[{year}] Raw data: {len(df):,} rows × {len(df.columns)} columns")
        return df

    except Exception as exc:
        print(f"[{year}] EXTRACTION ERROR: {exc}")
        return None


# ============================================================
# FILTER — keep only ODA flows and target columns
# ============================================================

def filter_and_clean(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Apply our filtering rules to a raw CRS DataFrame:
      1. Keep only ODA flows (FlowCode == 11, or FlowName contains 'ODA')
      2. Select and rename to our standard column names (KEEP_COLUMNS)
      3. Add a Year column if it's missing

    The raw CRS file uses different column name conventions than the
    'dotStat' export — we match case-insensitively to be robust.

    Args:
        df:   Raw DataFrame as extracted from the ZIP
        year: Calendar year (used to backfill Year column if missing)

    Returns:
        Filtered, standardised DataFrame. May be empty if no ODA rows found.
    """
    # ---- Step 1: Normalise column names ----
    # Strip whitespace, preserve original case for data
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Build a lowercase → original-case lookup for case-insensitive matching
    col_lower = {c.lower(): c for c in df.columns}

    print(f"[{year}] Columns in file: {list(col_lower.keys())[:12]}...")

    # ---- Step 2: Filter to ODA flows only ----
    # OECD uses FlowCode 11 for Official Development Assistance.
    # We look for FlowCode first (numeric), then FlowName (text fallback).
    if "flowcode" in col_lower:
        flow_col = col_lower["flowcode"]
        # Convert to numeric in case it arrived as strings
        df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce")
        n_before = len(df)
        df = df[df[flow_col] == ODA_FLOW_CODE]
        print(f"[{year}] ODA filter (FlowCode=={ODA_FLOW_CODE}): {n_before:,} → {len(df):,} rows")

    elif "flowname" in col_lower:
        flow_col = col_lower["flowname"]
        n_before = len(df)
        df = df[df[flow_col].str.contains("ODA", case=False, na=False)]
        print(f"[{year}] ODA filter (FlowName contains 'ODA'): {n_before:,} → {len(df):,} rows")

    else:
        print(f"[{year}] WARNING: No FlowCode or FlowName column — keeping all {len(df):,} rows")
        print(f"[{year}]          All columns: {list(df.columns)}")

    if len(df) == 0:
        print(f"[{year}] ERROR: No rows after ODA filter. Check column names above.")
        return pd.DataFrame()

    # ---- Step 3: Select and rename to our standard column names ----
    # We match case-insensitively to survive OECD API version changes
    rename_map   = {}   # actual column name → our standard name
    select_cols  = []   # list of actual column names to keep
    missing_cols = []   # our standard names that couldn't be found

    for standard_name in KEEP_COLUMNS:
        if standard_name.lower() in col_lower:
            actual = col_lower[standard_name.lower()]
            rename_map[actual]  = standard_name
            select_cols.append(actual)
        else:
            missing_cols.append(standard_name)

    if missing_cols:
        print(f"[{year}] WARNING: Columns not found in raw data: {missing_cols}")

    if not select_cols:
        print(f"[{year}] ERROR: None of our target columns found. Returning raw data for inspection.")
        return df

    # Apply column selection and rename
    result = df[select_cols].rename(columns=rename_map)

    # ---- Step 4: Ensure Year column is set ----
    # Some year files don't have a Year column (it's implicit from the filename)
    if "Year" not in result.columns:
        result["Year"] = year
        print(f"[{year}] Added Year column = {year}")
    else:
        # Coerce to integer, fill gaps with the expected year
        result["Year"] = pd.to_numeric(result["Year"], errors="coerce").fillna(year).astype(int)

    print(f"[{year}] Filtered: {len(result):,} ODA rows kept")
    return result


# ============================================================
# PER-YEAR ORCHESTRATION
# ============================================================

def process_one_year(year: int) -> Optional[pd.DataFrame]:
    """
    Full pipeline for one year:
      1. Return cached interim CSV if it already exists
      2. Download ZIP from OECD.Stat
      3. Extract CSV from ZIP
      4. Filter to ODA + target columns
      5. Save interim CSV to data/raw/oecd_crs_{year}_oda.csv

    Args:
        year: Calendar year to process

    Returns:
        Filtered DataFrame, or None if any step failed.
    """
    interim = _interim_path(year)

    # ---- Check for cached interim file ----
    if interim.exists() and interim.stat().st_size > 0:
        mb = interim.stat().st_size / 1_048_576
        print(f"[{year}] Loading cached file ({mb:.1f} MB): {interim.name}")
        return pd.read_csv(interim, low_memory=False)

    # ---- Download ----
    zip_bytes = download_year_zip(year)
    if zip_bytes is None:
        return None

    # ---- Extract ----
    raw_df = extract_csv_from_zip(zip_bytes, year)
    if raw_df is None or len(raw_df) == 0:
        return None

    # ---- Filter ----
    clean_df = filter_and_clean(raw_df, year)
    if len(clean_df) == 0:
        return None

    # ---- Cache the filtered year ----
    # Save a small interim CSV so we don't need to re-download this year
    clean_df.to_csv(interim, index=False)
    mb = interim.stat().st_size / 1_048_576
    print(f"[{year}] Saved interim file ({mb:.1f} MB): {interim.name}")

    return clean_df


# ============================================================
# SUMMARY
# ============================================================

def _print_summary(df: pd.DataFrame):
    """Print a concise summary of the final combined dataset."""
    print("\n--- Summary ---")
    print(f"  Total rows :  {len(df):,}")
    print(f"  Columns    :  {list(df.columns)}")

    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique().astype(int))
        print(f"  Years      :  {years[0]}–{years[-1]}  ({len(years)} years)")
        missing = [y for y in YEARS if y not in years]
        if missing:
            print(f"  *** MISSING YEARS: {missing}")

    if "DonorName" in df.columns:
        print(f"  Donors     :  {df['DonorName'].nunique()} unique")
        usa = df["DonorName"].str.contains("United States|USA|USAID", case=False, na=False).sum()
        if usa > 0:
            print(f"  USA rows   :  {usa:,}  OK")
        else:
            print("  *** WARNING: No USA rows found")

    print("---------------\n")


# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def download_oecd_crs() -> Optional[pd.DataFrame]:
    """
    Main entry point. Download, filter, and combine OECD CRS data 2018-2024.

    Runs one year at a time. Already-completed years are skipped via
    interim CSV cache files in data/raw/. Only re-downloads what is missing.

    Returns:
        Combined and filtered DataFrame, or None if all years failed.
    """
    print("\n" + "=" * 65)
    print("OECD DAC CRS Data Download")
    print(f"Target:  {YEARS[0]}–{YEARS[-1]}  |  ODA flows only")
    print(f"Output:  {OUTPUT_FILE}")
    print("=" * 65)

    ensure_dirs()

    # ---- Use cached combined file if available ----
    if output_already_exists():
        df = pd.read_csv(OUTPUT_FILE, low_memory=False)
        _print_summary(df)
        return df

    # ---- Process each year ----
    all_frames: List[pd.DataFrame] = []
    failed_years: List[int] = []

    for year in YEARS:
        df_year = process_one_year(year)
        if df_year is not None and len(df_year) > 0:
            all_frames.append(df_year)
        else:
            failed_years.append(year)
            print(f"[{year}] FAILED — will be missing from output")

    if not all_frames:
        print("\nERROR: No data downloaded for any year.")
        print("Check your internet connection and try again.")
        return None

    if failed_years:
        print(f"\nWARNING: The following years failed to download: {failed_years}")
        print("Analysis covering partial years may undercount ODA.")

    # ---- Combine all years ----
    print("\nCombining all years...")
    combined = pd.concat(all_frames, ignore_index=True)

    # ---- Save combined output ----
    combined.to_csv(OUTPUT_FILE, index=False)
    mb = OUTPUT_FILE.stat().st_size / 1_048_576
    print(f"Saved: {OUTPUT_FILE.name} ({mb:.1f} MB)")

    _print_summary(combined)
    return combined


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    result = download_oecd_crs()
    if result is None:
        sys.exit(1)
