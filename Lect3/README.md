# Fintech Synthetic Dataset - Comprehensive Analysis Pipeline

## 📋 Overview

This project performs an **end-to-end data analysis pipeline** on the Fintech Synthetic Dataset (2025), generating insights into 12 major fintech companies across 10 business segments throughout the year 2025.

The analysis includes exploratory data analysis (EDA), financial metrics computation, static visualizations, interactive dashboards, company rankings, and segment analysis with a complete professional report.

## 📊 Dataset

**File:** `fintech_top_sintetico_2025.csv`

- **Size:** 144 rows × 20 columns
- **Time Period:** January 2025 - December 2025 (monthly data)
- **Companies:** 12 major fintech players
- **Segments:** 10 business segments
- **Metrics:** Payment volume, revenue, users, churn, unit economics, marketing spend, and more

### Key Variables

| Variable | Description | Unit |
|----------|-------------|------|
| `Month` | Date (YYYY-MM format) | Date |
| `Company` | Fintech company name | String |
| `Segment` | Business segment | String |
| `Region` | Geographic region | String |
| `TPV_USD_B` | Total Payment Volume | Billions USD |
| `Revenue_USD_M` | Monthly revenue | Millions USD |
| `ARPU_USD` | Average Revenue Per User | USD |
| `Users_M` | Active users | Millions |
| `NewUsers_K` | New users acquired | Thousands |
| `Churn_pct` | Customer churn rate | Percentage |
| `CAC_USD` | Customer Acquisition Cost | USD |
| `TakeRate_pct` | Transaction take rate | Percentage |
| `Marketing_Spend_USD_M` | Marketing investment | Millions USD |

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
pandas >= 1.3.0
numpy >= 1.20.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
plotly >= 5.0.0
scipy >= 1.7.0
```

### Installation

1. **Install required packages:**

```bash
pip install -r requirements.txt
```

2. **Ensure data files are in the same directory:**
   - `fintech_top_sintetico_2025.csv`
   - `fintech_top_sintetico_dictionary.json`

### Running the Analysis

```bash
python mercado_miguel_fintech_analysis.py
```

The script will:
- Load and validate the dataset
- Perform comprehensive EDA
- Generate static visualizations (PNG)
- Create interactive dashboards (HTML)
- Compute company rankings
- Analyze business segments
- Export summary report

**Execution Time:** ~10-15 seconds

**Output Directory:** `fintech_outputs/`

## 📁 Project Structure

```
.
├── mercado_miguel_fintech_analysis.py  # Main analysis script
├── fintech_top_sintetico_2025.csv      # Dataset
├── fintech_top_sintetico_dictionary.json # Data dictionary
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── fintech_outputs/                    # Generated outputs
    ├── 📊 Static Visualizations (PNG)
    │   ├── revenue_by_segment_trend.png
    │   ├── tpv_top_companies_trend.png
    │   ├── user_acquisition_trend.png
    │   ├── churn_heatmap.png
    │   ├── arpu_by_segment.png
    │   └── correlation_matrix.png
    │
    ├── 📈 Interactive Dashboards (HTML)
    │   ├── interactive_revenue_timeseries.html
    │   ├── interactive_tpv_by_segment.html
    │   ├── interactive_users_scatter.html
    │   ├── interactive_arpu_vs_cac.html
    │   ├── interactive_revenue_by_segment.html
    │   ├── interactive_latest_month_ranking.html
    │   ├── interactive_churn_trends.html
    │   ├── interactive_marketing_efficiency.html
    │   ├── interactive_company_rankings.html
    │   └── interactive_segment_revenue.html
    │
    ├── 📋 Data Exports
    │   └── company_rankings.csv
    │
    └── 📄 Reports
        └── summary_report.txt
```

## 📊 Analysis Sections

### 1. **Data Loading & Preparation** (Section 1)
- Load CSV dataset and data dictionary
- Parse temporal data (Month column)
- Validate file existence and data integrity
- Create output directory structure

### 2. **Exploratory Data Analysis** (Section 2)
- Dataset overview (shape, date range, unique values)
- Missing value analysis (% missing per column)
- Company and segment distribution
- Descriptive statistics for all numeric features
- Temporal range inspection

### 3. **Financial Metrics Analysis** (Section 3)
Key business indicators computed:
- **Revenue Analysis:** Total and average revenue by segment
- **TPV Analysis:** Payment volume by company and segment
- **ARPU:** Average revenue per user by segment
- **User Metrics:** Active users and new user acquisition
- **Churn Analysis:** Churn rates by company
- **Unit Economics:** Customer Acquisition Cost (CAC) analysis

### 4. **Static Visualizations** (Section 4)
High-quality PNG charts for offline analysis:
- Time-series revenue trends
- TPV company comparison
- User acquisition momentum
- Churn rate heatmap
- ARPU by segment
- Correlation matrix of financial metrics

### 5. **Interactive Visualizations** (Section 5)
10 interactive Plotly dashboards for exploration:
- Revenue trends by company (line chart)
- TPV by segment (stacked bar chart)
- User metrics scatter (bubble chart)
- ARPU vs CAC analysis
- Revenue distribution box plot
- Company rankings snapshot
- Churn trends (time-series)
- Marketing efficiency scatter
- Comprehensive company rankings table
- Segment revenue comparison

**Interactive Features:**
- Hover for detailed information
- Click to toggle series on/off
- Download PNG snapshot button
- Zoom and pan controls
- Responsive design

### 6. **Company Rankings** (Section 6)
Performance metrics aggregated by company:
- Total annual revenue
- Average monthly revenue
- Peak active users
- Total new users acquired
- Average ARPU
- Average churn rate
- Average CAC
- Total marketing spend

### 7. **Segment Analysis** (Section 7)
Deep dive into business segments:
- Revenue totals and averages
- Payment volume (TPV) by segment
- User metrics by segment
- Unit economics (ARPU, CAC)
- Churn patterns
- Take rate analysis

### 8. **Summary Report** (Section 8)
Comprehensive text report including:
- Executive summary with key metrics
- Market overview statistics
- Top 5 companies by revenue
- Per-segment analysis
- Key insights and trends
- List of all generated outputs

## 📈 Key Findings & Metrics

The analysis reveals:

| Metric | Value |
|--------|-------|
| **Total Annual Revenue** | $72.5B (avg) |
| **Peak Active Users** | 884.4M |
| **Average ARPU** | $5.47 USD |
| **Total Payment Volume** | $31,626B USD |
| **Average Churn Rate** | 1.41% |
| **New Users Acquired** | ~2,900K users |

### Top Companies
1. **Visa** - Payments Network
2. **Mastercard** - Payments Network
3. **Alipay** - Super App / Wallet
4. **WeChat Pay** - Super App / Wallet
5. **Revolut** - Neobank

### Top Segments by Revenue
1. **Payments Network** - $15.1B
2. **BNPL** - $13.5B
3. **Super App / Wallet** - $10.1B
4. **Payments Infrastructure** - $6.7B
5. **Neobank / Super App** - $5.8B

## 🔄 Data Quality

- **Overall Completeness:** 84.4%
- **Missing Values:**
  - Private_Valuation_USD_B: 75.0% (private companies)
  - Ticker: 30.6% (non-public companies)
  - Close_USD: 30.6% (limited trading data)
  - Users_M, NewUsers_K: 16.67% (strategic non-disclosure)

## 🛠️ Technical Implementation

### Libraries Used
- **pandas:** Data manipulation and aggregation
- **numpy:** Numerical computing
- **matplotlib & seaborn:** Static visualization
- **plotly:** Interactive dashboards
- **scipy:** Statistical analysis
- **pathlib:** File handling
- **json:** Data dictionary parsing

### Code Quality Features
- Comprehensive docstrings (8 sections)
- Type hints where applicable
- Error handling for missing files
- Data validation checks
- Configurable random seed for reproducibility
- Professional console output with progress indicators

## 💡 Usage Examples

### Example 1: View Company Rankings
```bash
# Rankings saved to:
cat fintech_outputs/company_rankings.csv
```

### Example 2: Explore Interactive Dashboards
```bash
# Open in browser (Mac):
open fintech_outputs/interactive_revenue_timeseries.html
open fintech_outputs/interactive_arpu_vs_cac.html
open fintech_outputs/interactive_company_rankings.html
```

### Example 3: Generate Custom Analysis
Edit the script to add:
```python
# Custom metric aggregation
custom_metric = df.groupby('Company')['Revenue_USD_M'].agg(['sum', 'mean', 'std'])
print(custom_metric)
```

## 📊 Visualization Guide

### When to Use Each Chart

| Chart | Best For |
|-------|----------|
| Revenue time-series | Tracking trends over time |
| TPV by segment (stacked) | Comparing segment contributions |
| Users scatter | Identifying user acquisition patterns |
| ARPU vs CAC | Assessing unit economics quality |
| Churn heatmap | Spotting retention issues |
| Company rankings | Competitive analysis |
| Correlation matrix | Finding feature relationships |

## 🔍 Troubleshooting

### "FileNotFoundError: Missing required files"
**Solution:** Ensure both CSV and JSON files are in the same directory as the script:
```bash
ls fintech_top_sintetico_*.* 
```

### "ValueError in Plotly visualization"
**Solution:** Script automatically handles NaN values in visualizations. If error persists, check data quality:
```python
python -c "import pandas as pd; df = pd.read_csv('fintech_top_sintetico_2025.csv'); print(df.isna().sum())"
```

### "ModuleNotFoundError: No module named 'plotly'"
**Solution:** Install requirements:
```bash
pip install -r requirements.txt
```

## 📚 Dependencies Version Notes

- **plotly:** Requires `>=5.0.0` for `include_plotlyjs='cdn'` parameter
- **pandas:** Requires `>=1.3.0` for modern datetime handling
- **Python:** Tested on Python 3.8+ (including Python 3.14)

## 📄 Output Files Description

### PNG Files (Static Visualizations)
- High resolution (300 DPI)
- Suitable for printing and presentations
- No interactivity but fully formed insights

### HTML Files (Interactive Dashboards)
- **Size:** 8-25 KB per file
- **Browser Compatible:** Chrome, Firefox, Safari, Edge
- **Features:** Zoom, pan, hover tooltips, download as PNG
- **Load Time:** <1 second per file
- **CDN-based:** Uses plotly.js from CDN (requires internet for full features)

### CSV Export
- Company-level aggregated metrics
- 8 performance dimensions
- Ready for further analysis in Excel/R/Python

### Text Report
- Executive summary format
- Key metrics and findings
- Growth trends and insights
- Complete file listing

## ✨ Features Highlights

✅ **Comprehensive Analysis** - 8 organized sections  
✅ **Multi-format Outputs** - PNG, HTML, CSV, TXT  
✅ **Interactive Exploration** - 10 Plotly dashboards  
✅ **Professional Documentation** - Complete docstrings  
✅ **Data Quality Validation** - Missing value analysis  
✅ **Business Insights** - Revenue, growth, retention metrics  
✅ **Time-series Analysis** - Monthly trends and patterns  
✅ **Correlation Analysis** - Feature relationships  
✅ **Company Rankings** - Multi-dimensional comparison  
✅ **Segment Breakdown** - Business line analysis  

## 📖 Notes for Instructors

This analysis demonstrates:
- **Data Pipeline Design:** Well-structured 8-section workflow
- **EDA Best Practices:** Comprehensive statistical summaries
- **Visualization Types:** Both static and interactive approaches
- **Financial Analytics:** Domain-specific metric computation
- **Data Export:** Multiple formats for different use cases
- **Professional Output:** Report generation and documentation

## 👨‍💻 Author & Version

**Project:** Fintech Synthetic Analysis Pipeline  
**Author:** Miguel Mercado  
**Version:** 1.0  
**Date:** February 2025  
**Python Version:** 3.8+  

## 📝 License & Usage

This analysis script is provided for educational purposes in the FDAA course (Semester 7).

---

**For questions or improvements, refer to the docstring documentation in the main script.**
