from fastapi import FastAPI, Query, Path, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import date
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# =============================
# Environment & Database Setup
# =============================
load_dotenv()

PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =============================
# Sector mapping for companies
# =============================
SECTOR_MAP = {
    "AIRTEL": "Telecommunication",
    "BHL": "Hospitality",
    "FDHB": "Finance",
    "FMBCH": "Finance",
    "ICON": "Construction",
    "ILLOVO": "Agriculture",
    "MPICO": "Construction",
    "NBM": "Finance",
    "NBS": "Finance",
    "NICO": "Finance",
    "NITL": "Finance",
    "OMU": "Finance",
    "PCL": "Investments",
    "STANDARD": "Finance",
    "SUNBIRD": "Hospitality",
    "TNM": "Telecommunication",
}


# =============================
# Pydantic Models
# =============================
class Company(BaseModel):
    ticker: str
    name: str
    sector: str
    date_listed: date


class CompanyDetail(BaseModel):
    company: Company
    total_records: int


class PriceData(BaseModel):
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int]


class PriceSummary(BaseModel):
    period_high: float
    period_low: float
    total_volume: int


class DailyPriceResponse(BaseModel):
    ticker: str
    count: int
    data: List[PriceData]


class RangePriceResponse(BaseModel):
    ticker: str
    summary: PriceSummary
    data: List[PriceData]


class LatestPriceItem(BaseModel):
    ticker: str
    latest_date: date
    latest_price: float
    previous_price: Optional[float]
    change: Optional[float]
    percentage_change: Optional[str]


class LatestPriceResponse(BaseModel):
    count: int
    data: List[LatestPriceItem]


# =============================
# FastAPI app initialization
# =============================
app = FastAPI(title="MSE Data API", version="1.0")


# =============================
# 1. GET /companies
# =============================
@app.get("/companies", response_model=List[Company])
def get_companies(
    sector: Optional[str] = Query(None, description="Filter companies by sector")
):
    """
    Retrieve all companies listed on the MSE.
    Optionally filter by sector.
    """
    query = "SELECT ticker, name, date_listed FROM counters"
    df = pd.read_sql(text(query), engine)

    # Map sector using predefined SECTOR_MAP
    df["sector"] = df["ticker"].map(SECTOR_MAP)

    if sector:
        df = df[df["sector"].str.lower() == sector.lower()]

    if df.empty:
        raise HTTPException(
            status_code=404, detail="No companies found for the given filter"
        )

    return df.to_dict(orient="records")


# =============================
# 2. GET /companies/{ticker}
# =============================
@app.get("/companies/{ticker}", response_model=CompanyDetail)
def get_company_details(ticker: str = Path(..., description="Stock ticker symbol")):
    """
    Retrieve detailed information for a specific company, including total price records.
    """
    # Fetch company info
    query_company = text(
        "SELECT ticker, name, date_listed FROM counters WHERE ticker = :ticker"
    )
    df_company = pd.read_sql(query_company, engine, params={"ticker": ticker.upper()})

    if df_company.empty:
        raise HTTPException(
            status_code=404, detail=f"Company with ticker '{ticker}' not found"
        )

    # Add sector
    df_company["sector"] = df_company["ticker"].map(SECTOR_MAP)
    company_info = df_company.iloc[0].to_dict()

    # Fetch total price records
    counter_id_query = text("SELECT counter_id FROM counters WHERE ticker = :ticker")
    counter_id = pd.read_sql(
        counter_id_query, engine, params={"ticker": ticker.upper()}
    )["counter_id"].iloc[0]
    total_records = pd.read_sql(
        text("SELECT COUNT(*) AS count FROM prices WHERE counter_id = :counter_id"),
        engine,
        params={"counter_id": counter_id},
    )["count"].iloc[0]

    return {"company": company_info, "total_records": int(total_records)}


# =============================
# 3. GET /prices/daily
# =============================
@app.get("/prices/daily", response_model=DailyPriceResponse)
def get_daily_prices(
    ticker: str = Query(..., description="Stock ticker symbol"),
    start_date: Optional[date] = Query(None, description="Start date in YYYY-MM-DD"),
    end_date: Optional[date] = Query(None, description="End date in YYYY-MM-DD"),
    limit: int = Query(100, le=1000, description="Maximum number of records to return"),
):
    """
    Retrieve daily price data for a specific ticker with optional date range filtering.
    Uses numpy to replace invalid values (NaN, Inf) with zero for JSON safety.
    """
    df_counter = pd.read_sql(
        text("SELECT counter_id FROM counters WHERE ticker = :ticker"),
        engine,
        params={"ticker": ticker.upper()},
    )
    if df_counter.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    counter_id = df_counter["counter_id"].iloc[0]

    query = """
        SELECT trade_date, open_mwk AS open, high_mwk AS high, low_mwk AS low,
               close_mwk AS close, volume
        FROM prices
        WHERE counter_id = :counter_id
    """
    params = {"counter_id": counter_id}
    if start_date:
        query += " AND trade_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND trade_date <= :end_date"
        params["end_date"] = end_date
    query += " ORDER BY trade_date DESC LIMIT :limit"
    params["limit"] = limit

    df = pd.read_sql(text(query), engine, params=params)
    df = df.replace([np.nan, np.inf, -np.inf], 0)

    return {
        "ticker": ticker.upper(),
        "count": len(df),
        "data": df.to_dict(orient="records"),
    }


# =============================
# 4. GET /prices/range
# =============================
@app.get("/prices/range", response_model=RangePriceResponse)
def get_price_range(
    ticker: str = Query(..., description="Stock ticker symbol"),
    year: int = Query(..., description="Year of interest"),
    month: Optional[int] = Query(None, description="Month (1-12), optional"),
):
    """
    Retrieve price data for a specific year or month, including summary statistics.
    Invalid numeric values are replaced with zero using numpy.
    """
    df_counter = pd.read_sql(
        text("SELECT counter_id FROM counters WHERE ticker = :ticker"),
        engine,
        params={"ticker": ticker.upper()},
    )
    if df_counter.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    counter_id = df_counter["counter_id"].iloc[0]

    query = """
        SELECT trade_date, open_mwk AS open, high_mwk AS high, low_mwk AS low,
               close_mwk AS close, volume
        FROM prices
        WHERE counter_id = :counter_id
          AND EXTRACT(YEAR FROM trade_date) = :year
    """
    params = {"counter_id": counter_id, "year": year}
    if month:
        query += " AND EXTRACT(MONTH FROM trade_date) = :month"
        params["month"] = month

    df = pd.read_sql(text(query), engine, params=params)
    if df.empty:
        raise HTTPException(
            status_code=404, detail="No price data found for the given range"
        )
    df = df.replace([np.nan, np.inf, -np.inf], 0)

    summary = PriceSummary(
        period_high=float(df["high"].max()),
        period_low=float(df["low"].min()),
        total_volume=int(df["volume"].sum()),
    )

    return {
        "ticker": ticker.upper(),
        "summary": summary,
        "data": df.to_dict(orient="records"),
    }


# =============================
# 5. GET /prices/latest
# =============================
@app.get("/prices/latest", response_model=LatestPriceResponse)
def get_latest_prices(
    ticker: Optional[str] = Query(None, description="Stock ticker symbol")
):
    """
    Retrieve latest available prices for one or all tickers.
    Calculates change and percentage change compared to previous trading day.
    """
    tickers = [ticker.upper()] if ticker else list(SECTOR_MAP.keys())
    result = []

    for tk in tickers:
        df_counter = pd.read_sql(
            text("SELECT counter_id FROM counters WHERE ticker = :ticker"),
            engine,
            params={"ticker": tk},
        )
        if df_counter.empty:
            continue

        counter_id = df_counter["counter_id"].iloc[0]
        df = pd.read_sql(
            text(
                """
                SELECT trade_date, close_mwk AS close
                FROM prices
                WHERE counter_id = :counter_id
                ORDER BY trade_date DESC
                LIMIT 2
            """
            ),
            engine,
            params={"counter_id": counter_id},
        )
        if df.empty:
            continue

        latest = df.iloc[0]
        previous_close = df.iloc[1]["close"] if len(df) > 1 else None
        change = (latest["close"] - previous_close) if previous_close else None
        change_pct = (change / previous_close * 100) if previous_close else None

        result.append(
            LatestPriceItem(
                ticker=tk,
                latest_date=latest["trade_date"],
                latest_price=float(latest["close"]),
                previous_price=float(previous_close) if previous_close else None,
                change=float(change) if change else None,
                percentage_change=f"{change_pct:.2f}%" if change_pct else None,
            )
        )

    if not result:
        raise HTTPException(status_code=404, detail="No latest prices found")

    return {"count": len(result), "data": result}
