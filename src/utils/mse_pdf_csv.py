# mse_data_extractor.py

import logging
import os
import re
import sys
from datetime import date, datetime, time
from fileinput import filename
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pdfplumber
import camelot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# GLOBAL VARIABLES
# ===============================================
# Month map (handles "Sep" and "Sept")
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
COUNTER_LIST = {
    "2021-2025": [
        "AIRTEL",
        "BHL",
        "CIPLA",
        "FDHB",
        "FMBCH",
        "ICON",
        "ILLOVO",
        "NBS",
        "NICO",
        "NITL",
        "OMU",
        "PCL",
        "STANDARD",
        "SUNBIRD",
        "TNM",
        "UNIVERSAL",
    ]
}

COLS = {
    "2021-2025": [
        "counter_id",
        "daily_range_high",
        "daily_range_low",
        "counter",
        "buy_price",
        "sell_price",
        "previous_closing_price",
        "today_closing_price",
        "volume_traded",
        "dividend_mk",
        "dividend_yield_pct",
        "earnings_yield_pct",
        "pe_ratio",
        "pbv_ratio",
        "market_capitalization_mkmn",
        "profit_after_tax_mkmn",
        "num_shares_issue",
    ]
}
cols = [
    "counter_id",
    "daily_range_high",
    "daily_range_low",
    "counter",
    "buy_price",
    "sell_price",
    "previous_closing_price",
    "today_closing_price",
    "volume_traded",
    "dividend_mk",
    "dividend_yield_pct",
    "earnings_yield_pct",
    "pe_ratio",
    "pbv_ratio",
    "market_capitalization_mkmn",
    "profit_after_tax_mkmn",
    "num_shares_issue",
]

CASE1_DATE = [
    "2017-11-20",
    "2017-12-27",
    "2017-09-14",
    "2018-01-08",
    "2018-01-09",
    "2018-06-11",
    "2018-02-16",
    "2018-05-17",
]
CASE2_DATE = [
    "2018-07-03",
    "2018-07-04",
    "2018-07-05",
    "2018-07-09",
    "2018-07-10",
    "2018-07-12",
    "2018-07-13",
    "2018-07-16",
    "2018-07-17",
    "2018-07-18",
    "2018-07-19",
    "2018-07-20",
    "2018-07-23",
    "2018-07-25",
    "2018-07-26",
    "2018-07-27",
    "2018-06-27",
]


def _mkdate(y, m, d):  # y,m,d may be str
    return date(int(y), int(m), int(d))


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _parse_date_str(s: str, day_first: bool = True):
    """Parse a date from free text. Returns datetime.date or None."""
    s = _norm_text(s)

    # 1) 5 September 2025 | 05 Sep 2025 | 5 Sept, 2025 | 5th September 2025
    m = re.search(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(20\d{2})\b", s
    )
    if m:
        d, mon, y = m.groups()
        mon_num = _MONTHS.get(mon.lower())
        if mon_num:
            return _mkdate(y, mon_num, d)

    # 2) September 5, 2025 | Sep 05 2025 | Sept 5th 2025
    m = re.search(
        r"(?i)\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b", s
    )
    if m:
        mon, d, y = m.groups()
        mon_num = _MONTHS.get(mon.lower())
        if mon_num:
            return _mkdate(y, mon_num, d)

    # 3) ISO-like: 2025-09-05 / 2025/09/05 / 2025.09.05
    m = re.search(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if m:
        y, mth, d = m.groups()
        try:
            return _mkdate(y, mth, d)
        except ValueError:
            pass

    # 4) Numeric: 05-09-2025 | 05/09/2025 | 5.9.2025
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b", s)
    if m:
        a, b, y = m.groups()
        # day-first by default (MSE style)
        d, mth = (a, b) if day_first else (b, a)
        try:
            return _mkdate(y, mth, d)
        except ValueError:
            pass

    return None


def extract_date_from_filename(filename):
    """
    Extract date from a PDF filename in various formats and return a datetime.date object.

    Supported formats:
    - Daily_Report_DD_Month_YYYY.pdf
    - mse-daily-DD-MM-YYYY.pdf
    - mse-daily-YYYY-MM-DD.pdf

    Parameters:
    -----------
    filename : str
        The filename to parse

    Returns:
    --------
    datetime.date or None
        The extracted date or None if no date could be parsed
    """
    filename = Path(filename).name

    # Format: Daily_Report_03_January_2023.pdf
    pattern1 = r"Daily_Report_(\d{1,2})_([A-Za-z]+)_(\d{4})\.pdf"
    match = re.search(pattern1, filename)
    if match:
        day, month_str, year = match.groups()
        month_num = _MONTHS.get(month_str.lower())
        print(month_num, day, year)
        if month_num:
            return date(int(year), month_num, int(day))

    # Format: mse-daily-DD-MM-YYYY.pdf
    pattern2 = r"mse-daily-(\d{2})-(\d{2})-(\d{4})\.pdf"
    match = re.search(pattern2, filename)
    if match:
        day, month, year = match.groups()
        return date(int(year), int(month), int(day))

    # Format: mse-daily-YYYY-MM-DD.pdf
    pattern3 = r"mse-daily-(\d{4})-(\d{2})-(\d{2})\.pdf"
    match = re.search(pattern3, filename)
    if match:
        year, month, day = match.groups()
        return date(int(year), int(month), int(day))

    # If no pattern matches, try using _parse_date_str as fallback
    extracted_date = _parse_date_str(filename)
    if extracted_date:
        return extracted_date

    return None


def _parse_time_str(s: str):
    """Parse a time from free text. Returns datetime.time or None."""
    s = _norm_text(s)

    # 12-hour with seconds or without (e.g., 02:39:52 pm, 2:39 pm)
    m = re.search(r"(?i)\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)\b", s)
    if m:
        hh, mm, ss, ap = m.groups()
        hh, mm, ss = int(hh), int(mm), int(ss or 0)
        ap = ap.lower()
        if hh == 12:
            hh = 0
        if ap == "pm":
            hh += 12
        try:
            return time(hh, mm, ss)
        except ValueError:
            return None

    # 24-hour with optional seconds (e.g., 14:39:52 or 14:39)
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))\b", s)
    if m:
        hh, mm, ss = map(int, m.groups())
        try:
            return time(hh, mm, ss)
        except ValueError:
            return None

    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", s)
    if m:
        hh, mm = map(int, m.groups())
        try:
            return time(hh, mm)
        except ValueError:
            return None

    return None


def extract_print_date_time(
    pdf_path: str | Path, search_pages: int = 2, day_first: bool = True
):
    """
    Extract ONLY the 'Print Date' and 'Print Time' from the PDF text.

    Returns
    -------
    {
      'date': datetime.date | None,
      'time': datetime.time | None,
      'raw_date': str | None,  # snippet matched after the label (if any)
      'raw_time': str | None
    }
    """
    pdf_path = Path(pdf_path)
    raw_date_snip = raw_time_snip = None
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        n = min(max(search_pages, 1), len(pdf.pages))
        # Concatenate small chunks (keeps label context)
        page_texts = []
        for i in range(n):
            page_texts.append(pdf.pages[i].extract_text() or "")
        text = "\n".join(page_texts)

    # Prefer labeled fields
    m = re.search(r"(?is)Print\s*Date\s*:?\s*([^\n\r]+)", text)
    if m:
        raw_date_snip = m.group(1)
    m = re.search(r"(?is)Print\s*Time\s*:?\s*([^\n\r]+)", text)
    if m:
        raw_time_snip = m.group(1)

    d = _parse_date_str(raw_date_snip) if raw_date_snip else _parse_date_str(text)
    t = _parse_time_str(raw_time_snip) if raw_time_snip else _parse_time_str(text)

    return {
        "date": d,
        "time": t,
        "raw_date": (raw_date_snip or None),
        "raw_time": (raw_time_snip or None),
    }


def to_numeric_clean(val):
    """
    Clean and convert a value to numeric:
    - None/empty -> NaN
    - (123.45) -> -123.45
    - remove commas
    """
    if val is None or val == "N/A":
        return np.nan
    val = str(val).strip()
    if val.lower() == "none" or val == "":
        return np.nan
    # Handle parentheses as negatives
    if val.startswith("(") and val.endswith(")"):
        val = "-" + val[1:-1]
    # Remove commas, asterisks, and (SB)
    val = val.replace(",", "")
    val = val.replace("*", "")
    val = val.replace("(SB)", "")
    try:
        return float(val)
    except ValueError:
        return np.nan


def clean_cell(x):
    if x is None:
        return None
    x = re.sub(r"\s+", " ", str(x)).strip()
    x = x.replace("–", "-").replace("—", "-")
    d = {"-": None}
    # Expand dictionary
    x = d[x] if x in d else x
    return x if x else None


def is_numericish(s: Optional[str]) -> bool:
    if s is None:
        return False
    s = str(s).strip().replace(",", "")
    return bool(re.fullmatch(r"[-+]?(\d+(\.\d+)?|\.\d+)(%?)", s))


def is_header_like(row: list) -> bool:
    """Header-like = many text cells, few numeric cells."""
    cells = [c for c in row if c is not None and str(c).strip() != ""]
    if not cells:
        return False
    num_numeric = sum(1 for c in cells if is_numericish(c))
    num_alpha = sum(1 for c in cells if re.search(r"[A-Za-z]", str(c)))
    return (num_alpha >= max(1, len(cells) // 4)) and (num_numeric / len(cells) <= 0.5)


def normalize_to_width(rows: list[list], width: int) -> list[list]:
    out = []
    for r in rows:
        r = list(r)
        if len(r) < width:
            r = r + [None] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        out.append(r)
    return out


def clean_columns(df, col):
    for c in df.columns:
        if c != col:  # leave counter as string
            df[c] = df[c].apply(to_numeric_clean)

    # keep only rows with at least one numeric
    mask = df.apply(
        lambda row: any(
            isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val)
            for val in row
        ),
        axis=1,
    )
    df = df[mask]
    return df


def _split_column(df, source_col, new_cols, max_splits=3):
    """
    Splits a string column into multiple new columns based on spaces.
    Fills missing parts with a blank string.
    """
    parts = df[source_col].apply(lambda x: str(x).split(" "))
    for i, col in enumerate(new_cols):
        df[col] = parts.apply(lambda x: x[i] if len(x) > i else " ")
    return df


def format_shape_14(df):
    # Remove last row
    df = _split_column(df, "col_0", ["col_00", "col_01", "col_02"])
    df = _split_column(df, "col_5", ["col_55", "col_51"])

    # Reassign main columns
    df["col_0"] = df["col_00"]
    df["col_5"] = df["col_55"]

    df = df[
        [
            "col_0",
            "col_01",
            "col_02",
            "col_1",
            "col_2",
            "col_3",
            "col_4",
            "col_5",
            "col_51",
            "col_6",
            "col_7",
            "col_8",
            "col_9",
            "col_10",
            "col_11",
            "col_12",
            "col_13",
        ]
    ]
    return df


def format_shape_15(df, col):
    df = df[:-1]
    df = clean_columns(df, col)
    df = df[df[col].notna()]

    cl = df.columns
    df["col_01"] = np.nan
    df["col_02"] = np.nan

    df = df[[cl[0], "col_01", "col_02"] + list(cl[1:15])]
    return df


def format_shape_16(df, trade_date):
    date_str = str(trade_date)

    if date_str in ["2018-08-10", "2018-08-06", "2018-07-02"]:
        df = _split_column(df, "col_0", ["col_00", "col_01", "col_02"])
        df["col_0"] = df["col_00"]

        df = df[
            [
                "col_0",
                "col_01",
                "col_02",
                "col_1",
                "col_2",
                "col_3",
                "col_4",
                "col_5",
                "col_6",
                "col_7",
                "col_9",
                "col_10",
                "col_11",
                "col_12",
                "col_13",
                "col_14",
                "col_15",
            ]
        ]

        col = "col_1"
        df = clean_columns(df, col)
        df = df[df[col].notna()]

        if date_str in ["2018-08-10", "2018-08-06"]:
            df["col_7"] = df["col_6"] + df["col_7"]
            df["col_6"] = df["col_5"].apply(lambda x: x.split()[1])
            df["col_5"] = df["col_5"].apply(lambda x: x.split()[0])

    elif date_str == "2018-06-26":
        df = _split_column(df, "col_0", ["col_00", "col_01"])
        df["col_0"] = df["col_00"]

        df = df[
            [
                "col_0",
                "col_01",
                "col_1",
                "col_2",
                "col_3",
                "col_4",
                "col_5",
                "col_6",
                "col_7",
                "col_8",
                "col_9",
                "col_10",
                "col_11",
                "col_12",
                "col_13",
                "col_14",
                "col_15",
            ]
        ]

        col = "col_2"
        df = clean_columns(df, col)
        df = df[df[col].notna()]

    else:
        df = _split_column(df, "col_7", ["col_77", "col_71"])
        df["col_7"] = df["col_77"]

        df = df[
            [
                "col_0",
                "col_1",
                "col_2",
                "col_3",
                "col_4",
                "col_5",
                "col_6",
                "col_7",
                "col_71",
                "col_8",
                "col_9",
                "col_10",
                "col_11",
                "col_12",
                "col_13",
                "col_14",
                "col_15",
            ]
        ]

        col = "col_3"
        df = clean_columns(df, col)
        df = df[df[col].notna()]

    return df


def format_shape_19(df):
    df = _split_column(df, "col_0", ["col_00", "col_01", "col_02"])
    df["col_0"] = df["col_00"]

    df = df[
        [
            "col_0",
            "col_01",
            "col_02",
            "col_1",
            "col_2",
            "col_3",
            "col_4",
            "col_5",
            "col_6",
            "col_7",
            "col_8",
            "col_9",
            "col_10",
            "col_11",
            "col_12",
            "col_13",
            "col_14",
            "col_15",
            "col_16",
            "col_17",
            "col_18",
        ]
    ]
    return df


def combine_columns(row):
    n = len(str(row["col_18"])) - 2
    i = 10**n
    return float(row["col_17"]) * i + float(row["col_18"])


def reshape_dataframe(df, col):
    # Remove last 4 rows
    df = df[:-4]

    # Clean columns
    df = clean_columns(df, col)
    df = df[df[col].notna()]
    df = df.dropna(axis=1, how="all")

    # Add empty columns
    df["col_0"] = np.nan
    df["col_1"] = np.nan
    df["col_10"] = np.nan
    df["col_11"] = np.nan

    # Reorder columns
    cl = df.columns
    df = df[
        [
            "col_0",
            "col_1",
            cl[0],
            cl[1],
            cl[2],
            cl[3],
            "col_10",
            "col_11",
            cl[4],
            cl[5],
            cl[6],
            cl[7],
            cl[8],
            cl[9],
            cl[10],
            cl[11],
        ]
    ]
    return df.reset_index()


def parse_pdf_table_stream(pdf_path, col, date):
    tables = camelot.read_pdf(pdf_path, pages="1", flavor="stream")
    df = tables[0].df.reset_index(drop=True)
    df.columns = [f"col_{n}" for n in df.columns]

    # Clean columns
    df = clean_columns(df, col)
    df = df[df[col].notna()]

    if str(date) in ["2018-07-20", "2018-07-10"]:
        del df["col_9"]
        df["col_8"].fillna(value=0, inplace=True)

    elif str(date) == "2018-07-26":
        df["col_2"] = df["col_1"]
        df.iat[-2, 3] = "TNM"

    # Remove last row
    return df[:-1]


def processformat_shape_15(df, col, print_date):
    df = format_shape_15(df, col)
    if str(print_date) == "2018-08-13":
        df["col_10"] = df["col_9"]
        df["col_9"] = df["col_8"]
        df["col_8"] = df["col_7"]
        df["col_7"] = df["col_6"]
        df["col_6"] = df["col_5"].apply(lambda x: x.split()[1])
        df["col_5"] = df["col_5"].apply(lambda x: x.split()[0])
    if str(print_date) in [
        "2018-06-25",
        "2018-06-22",
        "2018-06-21",
        "2018-06-20",
        "2018-06-19",
        "2018-06-18",
    ]:

        def mapper(row):
            if len(str(row["col_6"]).split()) == 1:
                return row["col_7"]
            else:
                return str(row["col_6"]).split()[1] + str(row["col_7"])

        df["col_7"] = df.apply(mapper, axis=1)
        df["col_6"] = df["col_6"].apply(lambda x: float(str(x).split()[0]))
    return df


def extract_first_table(
    pdf_path: str | Path,
    out_dir: Optional[str | Path] = None,
) -> pd.DataFrame:

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try a few strategies to find tables
            strategies = [
                dict(
                    vertical_strategy="lines",
                    horizontal_strategy="lines",
                    snap_tolerance=3,
                    join_tolerance=3,
                    edge_min_length=3,
                ),
                dict(
                    vertical_strategy="lines_strict", horizontal_strategy="lines_strict"
                ),
                dict(vertical_strategy="text", horizontal_strategy="text"),
            ]
            tables = []
            for ts in strategies:
                try:
                    t = page.extract_tables(table_settings=ts) or []
                    for raw in t:
                        if raw and len(raw) >= 2 and max(len(r) for r in raw) >= 2:
                            tables.append(raw)
                    if tables:
                        break
                except Exception:
                    continue

            if not tables:
                continue
            # Use the first table found
            raw = tables[0]
            rows = [[clean_cell(c) for c in row] for row in raw]
            rows = [r for r in rows if any(c for c in r)]
            # Build DataFrame
            df = pd.DataFrame(rows).dropna(how="all")
            df.columns = [f"col_{n}" for n in df.columns]

            info = extract_print_date_time(pdf_path)
            print_date = info["date"]
            print_time = info["time"]
            print(df.shape)
            col = "col_3"
            if str(print_date) == "2018-06-14":
                df = df[:-1]
                df = clean_columns(df, col)
                df = df[df[col].notna()]
                df = df.dropna(axis=1, how="all")
            elif str(print_date) in CASE1_DATE:
                col = "col_2"
                df = reshape_dataframe(df, col)
            elif str(print_date) in CASE2_DATE:
                df = parse_pdf_table_stream(pdf_path, col, print_date)
            elif df.shape[1] == 14:
                df = format_shape_14(df)
                col = "col_1"
                df = clean_columns(df, col)
                df = df[df[col].notna()]
            elif df.shape[1] == 15:
                col = "col_1"
                df = processformat_shape_15(df, col, print_date)
            elif df.shape[1] == 16:
                df = format_shape_16(df, print_date)
            elif df.shape[1] == 18:
                col = "col_3"
                df = clean_columns(df, col)
                df = df[df[col].notna()]
                df = df.dropna(axis=1, how="all")
                if str(print_date) == "2018-08-07":
                    df["col_9"] = df["col_8"] + df["col_9"]
                    df["col_8"] = df["col_7"].apply(lambda x: x.split()[1])
                    df["col_7"] = df["col_7"].apply(lambda x: x.split()[0])
            elif df.shape[1] == 19:
                df = format_shape_19(df)
                col = "col_1"
                df = clean_columns(df, col)
                df = df[df[col].notna()]
                df = df.dropna(axis=1, how="all")
                df["col_17"] = df.apply(combine_columns, axis=1)
                del df["col_18"]
            elif df.shape[1] == 28:
                col = "col_2"
                df = df[:-4]
                df = clean_columns(df, col)
                df = df.reset_index()
                df = df[df[col].notna()]
                df = df.dropna(axis=1, how="all")
            elif df.shape[1] == 33:
                col = "col_4"
                df = df[:-2]
                df = clean_columns(df, col)
                df = df.reset_index()
                df = df[df[col].notna()]
                df = df.dropna(axis=1, how="all")
            elif df.shape[1] == 30 or df.shape[1] == 31:
                col = "col_2"
                df = df[:-4]
                print(df.shape)
                df = clean_columns(df, col)
                df = df.reset_index()
                df = df[df[col].notna()]
                df = df.dropna(axis=1, how="all")
            else:
                df = clean_columns(df, col)
                df = df[df[col].notna()]
                if str(print_date) == "2018-08-08":
                    df["col_9"] = df["col_8"]
                    df["col_8"] = df["col_7"].apply(lambda x: x.split()[1])
                    df["col_7"] = df["col_7"].apply(lambda x: x.split()[0])

            df.columns = cols
            df["trade_date"] = print_date
            df["print_time"] = print_time
            # Create CSV file based on date
            out_csv = out_dir / f"mse-daily-{print_date}.csv"

            if out_dir:
                df.to_csv(out_csv, index=False)
                print(f"✅ First table extracted and saved to {out_csv}")
                return out_csv
            return df

    print("⚠️ No table found in PDF.")
    return pd.DataFrame()


def get_most_recent_mse_report(directory_path):
    """
    Find the most recent MSE daily report PDF in a directory.

    Matches any PDF with date patterns like:
    - mse-daily-09/05/2025.pdf
    - mse-daily-09-05-2025.pdf
    - mse-daily-09_05_2025.pdf
    - mse_report_20250905.pdf
    - daily_report_2025-09-05.pdf
    etc.

    Args:
        directory_path (str): Path to directory containing MSE reports

    Returns:
        str: Path to the most recent PDF file, or None if no valid files found
    """
    try:
        directory = Path(directory_path)

        if not directory.exists():
            return None

        # More flexible patterns to match various date formats
        date_patterns = [
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",  # MM/DD/YYYY or MM-DD-YYYY
            r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",  # YYYY/MM/DD or YYYY-MM-DD
            r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
            r"(\d{1,2})_(\d{1,2})_(\d{4})",  # MM_DD_YYYY
            r"(\d{4})_(\d{1,2})_(\d{1,2})",  # YYYY_MM_DD
        ]

        pdf_files = []

        # Find all PDF files and try to extract dates
        for pdf_file in directory.glob("*.pdf"):
            print(f"Checking file: {pdf_file.name}")
            file_date = None

            for pattern in date_patterns:
                match = re.search(pattern, pdf_file.name)
                if match:
                    groups = match.groups()

                    try:
                        # Try different date interpretations
                        if len(groups[0]) == 4:  # Year first (YYYY-MM-DD)
                            year, month, day = (
                                int(groups[0]),
                                int(groups[1]),
                                int(groups[2]),
                            )
                        elif len(groups[2]) == 4:  # Year last (MM-DD-YYYY)
                            month, day, year = (
                                int(groups[0]),
                                int(groups[1]),
                                int(groups[2]),
                            )
                        else:  # YYYYMMDD format
                            year_str = groups[0]
                            if len(year_str) == 8:  # YYYYMMDD
                                year = int(year_str[:4])
                                month = int(year_str[4:6])
                                day = int(year_str[6:8])
                            else:
                                continue

                        file_date = datetime(year, month, day)
                        break

                    except ValueError:
                        continue

            if file_date:
                pdf_files.append((file_date, pdf_file))

        if not pdf_files:
            return None

        # Sort by date and return the most recent
        pdf_files.sort(key=lambda x: x[0], reverse=True)
        most_recent_file = pdf_files[0][1]

        return str(most_recent_file)

    except Exception as e:
        print(f"Error finding most recent MSE report: {e}")
        return None


def process_multiple_pdfs(
    input_dir: Path,
    out_dir: Path,
    start_date: date,
    cols: List[str],
    logs_dir: Optional[str | Path] = None,
) -> List[Optional[Path]]:
    not_processed = []
    for pdf_path in input_dir.glob("*.pdf"):
        try:
            file_date = extract_date_from_filename(pdf_path)
            if not file_date:
                print(f"⚠️  Skipping (no date in filename): {pdf_path.name}")
                continue
            if file_date >= start_date:
                print(f"Processing {pdf_path.name} dated {file_date}")
                output_file = extract_first_table(pdf_path=pdf_path, out_dir=out_dir)
                if output_file:
                    print(f"✅ Successfully Processed {pdf_path.name} -> {output_file}")
                else:
                    print(f"❌ Failed to process {pdf_path.name}")
                    not_processed.append(pdf_path.name)
                    continue
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            output_file = None
            not_processed.append(pdf_path.name)

    # Write to file unprocessed PDF filenames
    if not_processed:
        DIR_LOGS = Path.cwd().parent.parent / "logs/unprocessed_daily_pdfs"
        log_file = DIR_LOGS / "unprocessed_daily_pdfs.txt"
        with open(log_file, "w") as f:
            for fname in not_processed:
                f.write(f"{fname}\n")
        print(f"Unprocessed PDF filenames written to {log_file}")


def process_latest_report(
    input_dir: Path, out_dir: Path, cols: List[str]
) -> List[Optional[Path]]:

    # Example usage:
    pdf_path = get_most_recent_mse_report(input_dir)
    print(f"Most recent report: {pdf_path}")

    if not Path(pdf_path).exists():
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)

    print(f"🔍 Extracting data from: {pdf_path}")

    # Extract first table and save to CSV
    output_file = extract_first_table(
        pdf_path=pdf_path,
        out_dir=out_dir,
    )

    if output_file:
        print(f"✅ Data extraction completed successfully")
        print(f"📁 CSV file ready for inspection: {output_file}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review the CSV file: {output_file}")
        print(f"   2. Load data: python mse_data_loader.py {output_file}")
    else:
        print("❌ Failed to save data to CSV")
        sys.exit(1)


def merge_csv_into_master(data_dir: Path, master_csv: Path):
    """
    Combine all daily CSV files in data_dir into a master CSV file.
    """
    all_files = sorted(data_dir.glob("mse-daily-*.csv"))
    if not all_files:
        print(f"No CSV files found in {data_dir}")
        return

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Loaded {file} with {len(df)} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not df_list:
        print("No valid data to combine")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # Remove duplicates based on counter_id and trade_date, keeping the last occurrence
    master_df.drop_duplicates(
        subset=["counter_id", "trade_date"], keep="last", inplace=True
    )

    # Sort by trade_date descending, then counter_id ascending
    master_df.sort_values(
        by=["trade_date", "counter_id"], ascending=[False, True], inplace=True
    )

    # Save to master CSV
    master_df.to_csv(master_csv, index=False)
    print(f"✅ Master CSV created at {master_csv} with {len(master_df)} unique records")


# function updated
def main(process_latest=True, start_date_str="2017-01-01"):
    """
    Main function to extract MSE data from PDF and save to CSV
    """

    # SET WORKING DIRECTORY TO SCRIPT LOCATION
    # script_dir = Path(__file__).parent.parent
    DIR_DATA = Path.cwd().parent.parent / "data"
    DIR_REPORTS_PDF = DIR_DATA / "raw_pdfs/mse-daily-reports"
    DIR_REPORTS_CSV = DIR_DATA / "csv_files/mse-daily-data"
    DIR_LOGS = Path.cwd().parent.parent / "logs/unprocessed_daily_pdfs"
    master_csv = DIR_DATA / "output_combined_data/combined_csv.csv"
    # Standard columns in MSE daily report: 2021
    cols = COLS["2021-2025"]
    if process_latest:
        # Process only the most recent report
        process_latest_report(DIR_REPORTS_PDF, DIR_REPORTS_CSV, cols)
    else:
        # Process all reports from a start date
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        print(f"Processing all reports from {start_date} onwards...")
        process_multiple_pdfs(
            DIR_REPORTS_PDF, DIR_REPORTS_CSV, start_date, cols, DIR_LOGS
        )
        merge_csv_into_master(DIR_REPORTS_CSV, master_csv)


if __name__ == "__main__":
    PROCESS_LATEST = False
    main(process_latest=PROCESS_LATEST)
