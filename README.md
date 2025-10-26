# Malawi Stock Exchange (MSE) Data API

Version: 1.0

Framework: FastAPI

Database: PostgreSQL

Language: Python 3.10+

## Overview

The MSE Data API provides programmatic access to historical and real-time market information from the Malawi Stock Exchange.

It allows users to retrieve:

. Company details

. Price history

. Trading summaries for all listed companies

. Market statistics

## Base URL

Local: http: https://localhost:8000

Public (Ngrok): https://erosional-glennie-premeditatingly.ngrok-free.dev

## API Documentation

**Swagger UI:** http://localhost:8000/docs

**ReDoc UI:** http://localhost:8000/redoc



# Getting Started
## 1. Clone the Repository
git clone https://github.com/MarieMediatriceIradukunda/mse-api-assignment.git

cd mse-api-assignment

## 2. Create Virtual Environment

python3 -m venv venv

source venv/bin/activate  

## 3. Install Dependencies

pip install -r requirements.txt

## 4. Run the API Locally

uvicorn api:app --reload


## 5. (Optional) Run Using Ngrok

ngrok http 8000

## Environment Variables

Create a .env file in the project root with your PostgreSQL database credentials:

#### PostgreSQL configuration

PGHOST=your_database_host

PGPORT=5432

PGDATABASE=your_database_name

PGUSER=your_database_user

PGPASSWORD=your_database_password

## Endpoints Overview

1. `/companies`

**Method:** `GET`  
**Description:** Retrieve all listed companies. You can optionally filter by sector.

**Example Response:**

```json
[
  {
    "ticker": "AIRTEL",
    "name": "AIRTEL MALAWI PLC",
    "sector": "Telecommunication",
    "date_listed": "2020-02-24"
  },
  {
    "ticker": "TNM",
    "name": "TELEKOM NETWORKS MALAWI PLC",
    "sector": "Telecommunication",
    "date_listed": "2008-11-03"
  }
]
```
2. `/companies/{ticker}`

**Method:** `GET`  
**Description:** Retrieve detailed information for a specific company, including total number of recorded price entries.

**Example Response:**

```json
{
  "company": {
    "ticker": "AIRTEL",
    "name": "AIRTEL MALAWI PLC",
    "sector": "Telecommunication",
    "date_listed": "2020-02-24"
  },
  "total_records": 1210
}
```

3. `/prices/daily`

**Method:** `GET`  
**Description:** Retrieve daily price data for a given company ticker, optionally filtered by date range.

**Example Response:**

```json
{
  "ticker": "AIRTEL",
  "count": 2,
  "data": [
    {
      "trade_date": "2024-02-02",
      "open": 12.69,
      "high": 0,
      "low": 0,
      "close": 55.1,
      "volume": 0
    },
    {
      "trade_date": "2024-02-01",
      "open": 12.69,
      "high": 0,
      "low": 0,
      "close": 55.1,
      "volume": 0
    }
  ]
}
```

4. `/prices/range`

**Method:** `GET`  
**Description:** Retrieve all prices for a given year (and optional month), plus summary statistics for that period.

**Example Response:**

```json
{
  "ticker": "AIRTEL",
  "summary": {
    "period_high": 45,
    "period_low": 0,
    "total_volume": 14971601
  },
  "data": [
    {
      "trade_date": "2024-09-30",
      "open": 12.69,
      "high": 45,
      "low": 45,
      "close": 45,
      "volume": 3932
    },
    {
      "trade_date": "2024-09-27",
      "open": 12.69,
      "high": 44.97,
      "low": 44.96,
      "close": 44.97,
      "volume": 162625
    },.......]
```


5. `/prices/latest`

**Method:** `GET`  
**Description:** Retrieve the most recent closing price for one or all companies, including daily change and percentage change.

**Example Response:**

```json
{
  "count": 1,
  "data": [
    {
      "ticker": "TNM",
      "latest_date": "2025-09-19",
      "latest_price": 31.35,
      "previous_price": 34.84,
      "change": -3.490000000000002,
      "percentage_change": "-10.02%"
    }
  ]
}
```


## Technologies Used

**FastAPI:** For building the APIs

**PostgreSQL:** For storing data

**SQLAlchemy:** For database interactions

**Pandas :** Data cleaning and analysis

**Ngrok:** For public API exposure during development


# Malawi Stock Exchange API Assignment

## Overview
Build a comprehensive REST API to serve Malawi Stock Exchange (MSE) historical data. You'll process PDF reports spanning 2017-present, handle varying data formats, store data in PostgreSQL, and create API endpoints for data access.

## About Malawi Stock Exchange 
- **Website:** [https://mse.co.mw](https://mse.co.mw)
- **Daily Reports:** The website provides daily reports in PDF format [here](https://mse.co.mw/market/reports).
- **Listed Companies:** You can find a complete list of all companies (also referred to as counters) listed on the exchange [here](https://mse.co.mw/listing/mainboard). Please ensure you collect information for all listed companies by navigating through all pages.

## Timeline (4 Weeks)

### Week 1: Data Extraction & Processing (Sept 22-28)
**Focus**: Master PDF-to-CSV pipeline
- [ ] Analyze PDF report formats from 2017-present
- [ ] Update `mse_pdf2csv.py` to handle all report formats
- [ ] Process all historical reports to extract daily prices data into CSV files
- [ ] Generate clean, standardized CSV files

**Deliverable**: Working extraction notebook + processed CSV files. 

### Week 2: Database Design & Implementation (Sept 29 - Oct 5)
**Focus**: PostgreSQL setup and data management
- [ ] Design optimal database schema for historical data
- [ ] Create tables and relationships
- [ ] Implement data loading mechanisms

**Deliverable**: Database with all historical data loaded

### Week 3: API Development (Oct 6-12)
**Focus**: Build REST API endpoints
- [ ] Implement all 3-5 required API endpoints
- [ ] Add input validation using Pydantic models
- [ ] Add query parameters for filtering

**Deliverable**: Complete working API with all endpoints

### Week 4: Documentation, Testing & Polish (Oct 13-19)
**Focus**: Professional finishing touches
- [ ] Write comprehensive API documentation
- [ ] Prepare screenshots and demo materials
- **Optional**: Deploy API using ngrok for live evaluation

**Deliverable**: Production-ready API with full documentation and demo materials

**Final Submission Deadline: October 20, 2025**

## Data Source
**PDF Reports**: Download all MSE historical reports from [Google Drive](https://drive.google.com/drive/folders/1Bg7MiCViXxpPIB3jfpkClUM1HiLVrAEn?usp=sharing)

**Important**: Download all PDF reports and place them in the `data/raw_pdfs/` folder before starting the assignment.

## Setup
1. Clone this repository
2. Download PDF reports from the Google Drive link above
3. Place all PDFs in the `data/raw_pdfs/` folder
4. Create other necessary folders such as `csv_files`
5. Install dependencies: `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and configure database connection

## Assignment Tasks

### Task 1: PDF Data Extraction (25%)
**Objective**: Modify the existing `mse_pdf2csv.py` script to process all MSE reports from 2017 to present. Each report contains daily stock prices in tabular format - your task is to extract these complete daily price tables and convert them to CSV format. Reference the [example CSV files](https://drive.google.com/drive/folders/1-XTEDGa982HmXqpk92w5Du3oIHTvCMIq?usp=sharing) from our class exercises to understand the expected output structure and format. 

**Current Status**: The existing script in `src/utils/mse_pdf2csv.py` works for reports up to 2021. Your task is to modify/extend it to handle all reports through the present.

**Challenges to Address**:
- **Varying number of tickers** across different time periods
- **Format changes** in report tables post-2021
- **Inconsistent table structures** between older and newer reports
- **Data quality issues** and missing values. 

**Deliverables**:
- [ ] Updated `mse_pdf2csv.py` script that handles all report formats (2017-present)
- [ ] Error handling for problematic PDFs
- [ ] Data validation and cleaning
- [ ] Daily CSV files in a folder withing the data folder with standardized format
- [ ] Recommended: merged csv file for use in next step


### Task 2: Database Management (15%)
**Objective**: Set up and populate a PostgreSQL database with your extracted stock price data. You may build upon the database created during our class exercise. Use the same database schema and ticker information sources as outlined in our class exercise - refer to this [class document](https://docs.google.com/document/d/1w3N9B7hw41iS5cheyZYfnVUVdqLGiOmn-94eDJ4jRCQ/edit?usp=sharing) for the complete database structure and ticker table specifications.

**Requirements**:
- [ ] Create a database, you can call it `mse_db` or any other name
- [ ] Add two tables, one for `tickers` (as we did in class exercise) and another for `daily prices`
- [ ] Make sure you allow missing values in some columns as necessary 
- [ ] Handle duplicate records and data conflicts
- [ ] Load/import data into the database tables


### Task 3: API Development (20%)
**Objective**: Complete the REST API endpoints. Your API must implement the following 5 endpoints:
#### 1. GET /companies
**Description**: Return all companies listed on the MSE

**Query Parameters**:
- `sector` (optional): Filter by sector

**Example**: `GET /companies?sector=Banking`

**Response**: List of companies with ticker, name, sector, listing date

#### 2. GET /companies/{ticker}
**Description**: Get detailed information about a specific company

**Path Parameters**:
- `ticker` (required): Stock ticker symbol (e.g., "NICO")

**Example**: `GET /companies/NICO`

**Response**: Company details including ticker, full name, sector, listing date, description, and total records count

#### 3. GET /prices/daily
**Description**: Get daily stock prices with date filtering

**Query Parameters**:
- `ticker` (required): Stock ticker symbol
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)
- `limit` (optional): Maximum records to return (default: 100, max: 1000)

**Example**: `GET /prices/daily?ticker=NICO&start_date=2024-01-01&end_date=2024-12-31`

**Response**: Daily price data with open, high, low, close, volume, and trades


#### 4. GET /prices/range
**Description**: Get price data by month or year

**Query Parameters**:
- `ticker` (required): Stock ticker symbol
- `year` (required): Year (e.g., 2024)
- `month` (optional): Month (1-12). If omitted, returns entire year

**Example**: `GET /prices/range?ticker=NICO&year=2024&month=6`

**Response**: All daily prices for the specified period, plus summary statistics (period high, low, total volume, etc.)


#### 5. GET /prices/latest
**Description**: Get the most recent trading day prices

**Query Parameters**:
- `ticker` (optional): Stock ticker symbol. If omitted, returns latest prices for all stocks

**Example**: `GET /prices/latest?ticker=NICO`

**Response**: Latest available price data including change and change percentage from previous day

#### API Requirements
- [ ] Use **Pydantic models** for request validation and response schemas
- [ ] Return proper **HTTP status codes** (200, 400, 404, 422, 500)
- [ ] Include **error handling** with descriptive error messages
- [ ] All dates must use **YYYY-MM-DD** format
- [ ] Test all endpoints in FastAPI's interactive docs (`/docs`)

### Task 4: Project Demo & Documentation (40%)
**Objective**:
Showcase your functional API using detailed visual documentation. Create a `screenshots` folder inside the `docs` directory where instructors can review your demonstration materials as outlined in the requirements below.

**Required Screenshots** (submit in `screenshots/` folder):
- [ ] **Startup**: Terminal showing `uvivorn main:app --reload` running successfully
- [ ] **API Home**: Browser showing API running at `http://localhost:8000`
- [ ] **Interactive Docs**: FastAPI docs at `http://localhost:8000/docs`
- [ ] **Endpoint Testing**: Each of the 3-5 endpoints returning sample data
- [ ] **Database Connection**: Evidence of successful database connection (pgAdmin/terminal)

**Screenshot Requirements**:
- High-quality, clear images showing full browser/terminal windows
- Include timestamps or other proof of recency
- Name files descriptively: `01_api_startup.png`, `02_interactive_docs.png`, etc.


## Bonus Tasks (Optional-Additional Points)

### Live API Demo via ngrok (10% points)
**For Advanced Students**: Deploy your API using ngrok for live instructor testing

**Requirements**:
- [ ] Install and configure ngrok
- [ ] Deploy API with public URL
- [ ] Submit ngrok URL for real-time evaluation
- [ ] Schedule brief demo session with instructor
- [ ] Ensure API remains accessible during evaluation period

**Submission**: Include ngrok URL and availability schedule in your final submission

## Evaluation Criteria & Grading

| Component | Points | Evaluation Method |
|-----------|--------|-------------------|
| **PDF Processing** | 25% | Code review + output quality |
| **Database Design** | 15% | Schema design + data loading success |
| **API Implementation** | 20% | Code quality + endpoint functionality |
| **Project Documentation** | 40% | **Screenshot demonstration** |
| **Bonus: Live Demo** | +10% | **Real-time ngrok testing** |

## Critical Success Criteria
**To receive full credit, your project MUST:**
1. Process all PDF reports from 2017-present successfully
2. Load data into PostgreSQL database with proper schema
3. Implement all 3-5 API endpoints with proper validation
4. **Provide clear screenshots demonstrating full functionality**


## Submission Guidelines
1. **Code**: Push all code to your GitHub repository. This is what will be used as your submission. Instructors will clone your repository on the final submission day.
2. **Screenshots**: Include `screenshots/` folder with all required images
3. **Documentation**: Update README with your specific setup instructions
4. **Database**: Include SQL schema and sample data files
5. **Optional**: Submit ngrok URL for bonus evaluation

**Final Submission Deadline: October 20, 2025**

## Technical Requirements
- Use FastAPI for API development
- PostgreSQL for data storage
- Pandas and other packages for data processing
- Clean, documented code
- Clear visual documentation via screenshots

---
