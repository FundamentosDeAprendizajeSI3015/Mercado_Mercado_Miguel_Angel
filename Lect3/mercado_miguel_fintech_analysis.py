# fintech_analysis_interactive.py
# -*- coding: utf-8 -*-
"""
================================================================================
COMPREHENSIVE FINTECH DATASET ANALYSIS WITH INTERACTIVE VISUALIZATIONS
================================================================================

This module performs an end-to-end data analysis pipeline on the Fintech
Synthetic Dataset (2025):
  • Exploratory Data Analysis (EDA) with time-series insights
  • Financial metrics analysis (TPV, Revenue, ARPU, Churn, CAC)
  • Company and segment comparison
  • Correlation and trend analysis
  • Interactive visualizations exported to HTML using Plotly
  • Static visualizations using Matplotlib and Seaborn
  • Time-series decomposition and monthly trends

Execution:
    python fintech_analysis_interactive.py

Required packages:
    pandas, numpy, matplotlib, seaborn, plotly, scipy
    
Dataset Files (must be in same directory):
    - fintech_top_sintetico_2025.csv
    - fintech_top_sintetico_dictionary.json

Author: Data Analysis Pipeline
Date: 2025
================================================================================
"""

from __future__ import annotations
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# ============================================================================
# INTERACTIVE VISUALIZATIONS WITH PLOTLY
# ============================================================================
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_html
from plotly.subplots import make_subplots

# Set visual style and random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================================
"""
Load the Fintech Synthetic Dataset and prepare it for comprehensive analysis.
The dataset contains monthly metrics for 12 major fintech companies across
multiple segments (payments, neobanco, BNPL, crypto, etc.) for all of 2025.
"""

print("\n" + "="*80)
print("FINTECH DATASET ANALYSIS - COMPREHENSIVE DATA PIPELINE")
print("="*80)

# Load data dictionary for documentation
dict_path = Path('fintech_top_sintetico_dictionary.json')
csv_path = Path('fintech_top_sintetico_2025.csv')

if not csv_path.exists() or not dict_path.exists():
    raise FileNotFoundError("Missing required files. Ensure fintech_top_sintetico_2025.csv "
                          "and fintech_top_sintetico_dictionary.json are in this directory.")

print("\n[DATA LOADING] Reading dataset")
print("-" * 80)

# Load the data dictionary
with open(dict_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

# Load the CSV dataset
df = pd.read_csv(csv_path)

# Parse date column and ensure temporal ordering
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.sort_values('Month').reset_index(drop=True)

print(f"Dataset shape: {df.shape} (rows, columns)")
print(f"Time period: {df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}")
print(f"Number of companies: {df['Company'].nunique()}")
print(f"Segments covered: {df['Segment'].nunique()}")

# Create output directory for results
os.makedirs("fintech_outputs", exist_ok=True)

# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
"""
Perform comprehensive EDA including:
  • Dataset overview and statistics
  • Missing value analysis
  • Company and segment distribution
  • Key financial metrics summary
  • Temporal trends
"""

print("\n[EDA] Dataset Overview")
print("-" * 80)

print(f"\nDataset Info:")
print(f"  Total rows: {len(df)}")
print(f"  Date range: {(df['Month'].max() - df['Month'].min()).days} days")
print(f"  Unique companies: {df['Company'].nunique()}")
print(f"  Unique segments: {df['Segment'].nunique()}")
print(f"  Unique regions: {df['Region'].nunique()}")

print(f"\nCompanies analyzed:")
for i, company in enumerate(sorted(df['Company'].unique()), 1):
    print(f"  {i:2d}. {company}")

print(f"\nBusiness Segments:")
for segment in sorted(df['Segment'].unique()):
    count = df[df['Segment'] == segment].shape[0]
    print(f"  • {segment}: {count} records")

# Missing values analysis
print(f"\nMissing Values (%):")
missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]
if len(missing) > 0:
    for col, pct in missing.items():
        print(f"  {col:30s}: {pct:6.2f}%")
else:
    print("  No missing values detected ✓")

# Basic statistics
print(f"\nKey Metrics - Descriptive Statistics:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
summary_stats = df[numeric_cols].describe().round(2)
print(summary_stats.to_string())

# ============================================================================
# SECTION 3: FINANCIAL METRICS ANALYSIS
# ============================================================================
"""
Analyze key fintech business metrics:
  • Total Payment Volume (TPV) trends
  • Revenue and profitability
  • Average Revenue Per User (ARPU)
  • Customer Acquisition Cost (CAC)
  • Churn rate trends
  • User acquisition momentum
"""

print("\n[FINANCIAL METRICS] Analyzing Key Indicators")
print("-" * 80)

# Revenue analysis
df['Revenue_USD_M'] = pd.to_numeric(df['Revenue_USD_M'], errors='coerce')
print(f"\nRevenue by Segment (avg monthly, USD M):")
revenue_by_segment = df.groupby('Segment')['Revenue_USD_M'].mean().sort_values(ascending=False)
for segment, revenue in revenue_by_segment.items():
    print(f"  {segment:30s}: ${revenue:>10,.2f}M")

# TPV analysis
df['TPV_USD_B'] = pd.to_numeric(df['TPV_USD_B'], errors='coerce')
print(f"\nTotal Payment Volume by Company (avg monthly, USD B):")
tpv_by_company = df.groupby('Company')['TPV_USD_B'].mean().sort_values(ascending=False)
for company, tpv in tpv_by_company.head(10).items():
    print(f"  {company:25s}: ${tpv:>8.2f}B")

# ARPU analysis
df['ARPU_USD'] = pd.to_numeric(df['ARPU_USD'], errors='coerce')
print(f"\nAverage Revenue Per User (ARPU, monthly):")
arpu_by_segment = df.groupby('Segment')['ARPU_USD'].mean().sort_values(ascending=False)
for segment, arpu in arpu_by_segment.items():
    print(f"  {segment:30s}: ${arpu:>8.2f}")

# User metrics
df['Users_M'] = pd.to_numeric(df['Users_M'], errors='coerce')
df['NewUsers_K'] = pd.to_numeric(df['NewUsers_K'], errors='coerce')
total_users = df.groupby('Company')['Users_M'].max().sort_values(ascending=False)
print(f"\nActive Users by Company (peak 2025, millions):")
for company, users in total_users.head(10).items():
    print(f"  {company:25s}: {users:>8.1f}M users")

# CAC and Churn
df['CAC_USD'] = pd.to_numeric(df['CAC_USD'], errors='coerce')
df['Churn_pct'] = pd.to_numeric(df['Churn_pct'], errors='coerce')
print(f"\nChurn Rate (avg %):")
churn_by_company = df.groupby('Company')['Churn_pct'].mean().sort_values(ascending=False)
for company, churn in churn_by_company.head(10).items():
    print(f"  {company:25s}: {churn:>6.2f}%")

# ============================================================================
# SECTION 4: STATIC VISUALIZATIONS
# ============================================================================
"""
Create high-quality static visualizations for offline analysis:
  • Revenue and TPV trends
  • Segment comparison
  • Company performance metrics
  • Correlation matrices
"""

print("\n[VISUALIZATIONS] Generating Static Plots")
print("-" * 80)

# 4.1 Revenue trend by segment
fig, ax = plt.subplots(figsize=(14, 7))
df_pivot = df.groupby(['Month', 'Segment'])['Revenue_USD_M'].sum().unstack(fill_value=0)
df_pivot.plot(ax=ax, marker='o', linewidth=2)
ax.set_title('Monthly Revenue by Business Segment', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue (USD Millions)')
ax.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fintech_outputs/revenue_by_segment_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] fintech_outputs/revenue_by_segment_trend.png")

# 4.2 TPV by company over time
fig, ax = plt.subplots(figsize=(14, 7))
companies_top = df.groupby('Company')['TPV_USD_B'].sum().nlargest(6).index
df_top = df[df['Company'].isin(companies_top)]
for company in companies_top:
    data = df_top[df_top['Company'] == company].sort_values('Month')
    ax.plot(data['Month'], data['TPV_USD_B'], marker='o', label=company, linewidth=2)
ax.set_title('Total Payment Volume (TPV) - Top 6 Companies', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('TPV (USD Billions)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fintech_outputs/tpv_top_companies_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] fintech_outputs/tpv_top_companies_trend.png")

# 4.3 User acquisition momentum
fig, ax = plt.subplots(figsize=(14, 7))
df_users = df.groupby('Month')[['Users_M', 'NewUsers_K']].sum()
ax2 = ax.twinx()
ax.bar(df_users.index, df_users['NewUsers_K'], alpha=0.6, label='New Users', color='steelblue')
ax2.plot(df_users.index, df_users['Users_M'], color='darkred', marker='o', 
         linewidth=2.5, markersize=8, label='Total Active Users')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('New Users (thousands)', fontsize=11, color='steelblue')
ax2.set_ylabel('Total Active Users (millions)', fontsize=11, color='darkred')
ax.set_title('User Acquisition and Active Base Over 2025', fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='darkred')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('fintech_outputs/user_acquisition_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] fintech_outputs/user_acquisition_trend.png")

# 4.4 Churn rate heatmap by company
fig, ax = plt.subplots(figsize=(12, 8))
churn_pivot = df.pivot_table(values='Churn_pct', index='Company', columns='Month', aggfunc='mean')
sns.heatmap(churn_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Churn %'})
ax.set_title('Monthly Churn Rate Heatmap by Company', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Company')
plt.tight_layout()
plt.savefig('fintech_outputs/churn_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] fintech_outputs/churn_heatmap.png")

# 4.5 ARPU comparison by segment
fig, ax = plt.subplots(figsize=(12, 7))
segment_stats = df.groupby('Segment')['ARPU_USD'].agg(['mean', 'std', 'min', 'max']).sort_values('mean')
x = np.arange(len(segment_stats))
ax.barh(x, segment_stats['mean'], xerr=segment_stats['std'], capsize=5, alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(segment_stats.index)
ax.set_xlabel('ARPU (USD)')
ax.set_title('Average Revenue Per User (ARPU) by Segment', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('fintech_outputs/arpu_by_segment.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] fintech_outputs/arpu_by_segment.png")

# 4.6 Correlation matrix of numeric features
numeric_features = ['TPV_USD_B', 'Revenue_USD_M', 'ARPU_USD', 'Users_M', 
                   'NewUsers_K', 'Churn_pct', 'CAC_USD', 'Marketing_Spend_USD_M']
numeric_features = [col for col in numeric_features if col in df.columns]
corr_matrix = df[numeric_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix - Financial Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fintech_outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] fintech_outputs/correlation_matrix.png")

# ============================================================================
# SECTION 5: INTERACTIVE VISUALIZATIONS WITH PLOTLY
# ============================================================================
"""
Create interactive HTML visualizations for detailed exploration:
  • Time-series with hover details
  • Segment and company comparisons
  • Scatter plots showing relationships
  • Performance dashboards
"""

print("\n[INTERACTIVE PLOTS] Creating Plotly Visualizations")
print("-" * 80)

# 5.1 Revenue time-series with company breakdown
revenue_monthly = df.groupby(['Month', 'Company'])['Revenue_USD_M'].sum().reset_index()
fig_revenue = px.line(
    revenue_monthly,
    x='Month', y='Revenue_USD_M', color='Company',
    title='Monthly Revenue Trend by Company (2025)',
    labels={'Month': 'Month', 'Revenue_USD_M': 'Revenue (USD Millions)', 'Company': 'Company'},
    height=600
)
fig_revenue.update_layout(
    hovermode='x unified',
    xaxis_title='Month',
    yaxis_title='Revenue (USD Millions)',
    font=dict(size=11)
)
write_html(fig_revenue, 'fintech_outputs/interactive_revenue_timeseries.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_revenue_timeseries.html")

# 5.2 TPV by segment comparison
tpv_segment = df.groupby(['Month', 'Segment'])['TPV_USD_B'].sum().reset_index()
fig_tpv = px.bar(
    tpv_segment,
    x='Month', y='TPV_USD_B', color='Segment',
    title='Total Payment Volume (TPV) by Segment',
    labels={'Month': 'Month', 'TPV_USD_B': 'TPV (USD Billions)', 'Segment': 'Segment'},
    height=600,
    barmode='stack'
)
fig_tpv.update_layout(font=dict(size=11))
write_html(fig_tpv, 'fintech_outputs/interactive_tpv_by_segment.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_tpv_by_segment.html")

# 5.3 Users and new users scatter
df_clean = df.dropna(subset=['Users_M', 'NewUsers_K', 'Company'])
fig_users = px.scatter(
    df_clean,
    x='Users_M', y='NewUsers_K', color='Segment', size='Revenue_USD_M',
    hover_name='Company', hover_data=['Month', 'ARPU_USD', 'Churn_pct'],
    title='User Base vs New User Acquisition',
    labels={'Users_M': 'Active Users (Millions)', 'NewUsers_K': 'New Users (Thousands)'},
    height=600
)
fig_users.update_layout(font=dict(size=11))
write_html(fig_users, 'fintech_outputs/interactive_users_scatter.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_users_scatter.html")

# 5.4 ARPU vs CAC analysis
df_clean_cac = df.dropna(subset=['ARPU_USD', 'CAC_USD', 'Company', 'Users_M'])
fig_cac = px.scatter(
    df_clean_cac,
    x='ARPU_USD', y='CAC_USD', color='Segment', size='Users_M',
    hover_name='Company', hover_data=['Month', 'Churn_pct'],
    title='Unit Economics: ARPU vs Customer Acquisition Cost',
    labels={'ARPU_USD': 'ARPU (USD)', 'CAC_USD': 'CAC (USD)', 'Users_M': 'Active Users (M)'},
    height=600
)
fig_cac.update_layout(font=dict(size=11))
write_html(fig_cac, 'fintech_outputs/interactive_arpu_vs_cac.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_arpu_vs_cac.html")

# 5.5 Revenue and margins box plot by segment
fig_box = px.box(
    df,
    x='Segment', y='Revenue_USD_M',
    color='Segment',
    title='Revenue Distribution by Business Segment',
    labels={'Revenue_USD_M': 'Revenue (USD Millions)', 'Segment': 'Segment'},
    height=600
)
fig_box.update_layout(font=dict(size=11), showlegend=False)
write_html(fig_box, 'fintech_outputs/interactive_revenue_by_segment.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_revenue_by_segment.html")

# 5.6 Company performance dashboard (last month snapshot)
df_latest = df[df['Month'] == df['Month'].max()].sort_values('Revenue_USD_M', ascending=True)
fig_dash = px.bar(
    df_latest,
    x='Revenue_USD_M', y='Company',
    color='Segment',
    title=f'Company Revenue Ranking - {df_latest["Month"].iloc[0].strftime("%B %Y")}',
    labels={'Revenue_USD_M': 'Revenue (USD Millions)', 'Company': 'Company'},
    height=600,
    orientation='h'
)
fig_dash.update_layout(font=dict(size=11))
write_html(fig_dash, 'fintech_outputs/interactive_latest_month_ranking.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_latest_month_ranking.html")

# 5.7 Churn trends
churn_trend = df.groupby(['Month', 'Segment'])['Churn_pct'].mean().reset_index()
fig_churn = px.line(
    churn_trend,
    x='Month', y='Churn_pct', color='Segment',
    title='Average Churn Rate Trend by Segment',
    labels={'Month': 'Month', 'Churn_pct': 'Churn Rate (%)', 'Segment': 'Segment'},
    height=600,
    markers=True
)
fig_churn.update_layout(font=dict(size=11), hovermode='x unified')
write_html(fig_churn, 'fintech_outputs/interactive_churn_trends.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_churn_trends.html")

# 5.8 Marketing spend efficiency
df_clean_mkt = df.dropna(subset=['Marketing_Spend_USD_M', 'NewUsers_K'])
fig_mkt = px.scatter(
    df_clean_mkt,
    x='Marketing_Spend_USD_M', y='NewUsers_K',
    color='Segment', size='Revenue_USD_M',
    hover_name='Company', hover_data=['Month'],
    title='Marketing Spend Efficiency - Spend vs New Users',
    labels={'Marketing_Spend_USD_M': 'Marketing Spend (USD Millions)', 
            'NewUsers_K': 'New Users (Thousands)'},
    height=600
)
fig_mkt.update_layout(font=dict(size=11))
write_html(fig_mkt, 'fintech_outputs/interactive_marketing_efficiency.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_marketing_efficiency.html")

# ============================================================================
# SECTION 6: COMPARATIVE ANALYSIS - COMPANY RANKINGS
# ============================================================================
"""
Create summary metrics and rankings for companies across key dimensions:
  • Revenue leaders
  • Growth momentum
  • Unit economics quality
  • Customer retention
"""

print("\n[RANKINGS] Computing Company Performance Metrics")
print("-" * 80)

rankings = df.groupby('Company').agg({
    'Revenue_USD_M': ['sum', 'mean', 'max'],
    'TPV_USD_B': 'sum',
    'Users_M': 'max',
    'NewUsers_K': 'sum',
    'ARPU_USD': 'mean',
    'Churn_pct': 'mean',
    'CAC_USD': 'mean',
    'Marketing_Spend_USD_M': 'sum'
}).round(2)

rankings.columns = ['_'.join(col).strip() for col in rankings.columns.values]
rankings = rankings.sort_values('Revenue_USD_M_sum', ascending=False)

print("\nTop Companies by Total Annual Revenue:")
for i, (company, row) in enumerate(rankings.head(5).iterrows(), 1):
    print(f"  {i}. {company:25s} - ${row['Revenue_USD_M_sum']:>10,.2f}M (Avg ARPU: ${row['ARPU_USD_mean']:>6.2f})")

# Export rankings to CSV
rankings.to_csv('fintech_outputs/company_rankings.csv')
print("\n[SAVED] fintech_outputs/company_rankings.csv")

# Interactive rankings table
fig_rankings = go.Figure(data=[go.Table(
    header=dict(
        values=['Company', 'Total Revenue (M)', 'Avg Revenue (M)', 'Total TPV (B)', 
                'Peak Users (M)', 'Avg ARPU', 'Churn %', 'Avg CAC'],
        fill_color='steelblue',
        align='left',
        font=dict(color='white', size=11)
    ),
    cells=dict(
        values=[
            rankings.index,
            rankings['Revenue_USD_M_sum'].round(1),
            rankings['Revenue_USD_M_mean'].round(1),
            rankings['TPV_USD_B_sum'].round(2),
            rankings['Users_M_max'].round(1),
            rankings['ARPU_USD_mean'].round(2),
            rankings['Churn_pct_mean'].round(2),
            rankings['CAC_USD_mean'].round(2)
        ],
        fill_color='lavender',
        align='left',
        font=dict(size=10)
    )
)])
fig_rankings.update_layout(title_text='Company Performance Rankings (Full Year 2025)', height=600)
write_html(fig_rankings, 'fintech_outputs/interactive_company_rankings.html', include_plotlyjs='cdn')
print("[SAVED] fintech_outputs/interactive_company_rankings.html")

# ============================================================================
# SECTION 7: SEGMENT ANALYSIS
# ============================================================================
"""
Deep dive into business segment performance and characteristics
"""

print("\n[SEGMENT ANALYSIS] Segment-Level Metrics")
print("-" * 80)

segment_stats = df.groupby('Segment').agg({
    'Revenue_USD_M': ['sum', 'mean'],
    'TPV_USD_B': 'sum',
    'Users_M': ['sum', 'mean'],
    'ARPU_USD': 'mean',
    'Churn_pct': 'mean',
    'CAC_USD': 'mean',
    'TakeRate_pct': 'mean'
}).round(2)

segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns.values]
segment_stats = segment_stats.sort_values('Revenue_USD_M_sum', ascending=False)

print("\nSegment Performance Summary:")
for segment, row in segment_stats.iterrows():
    print(f"\n{segment}:")
    print(f"  Total Revenue: ${row['Revenue_USD_M_sum']:,.2f}M")
    print(f"  Total TPV: ${row['TPV_USD_B_sum']:.2f}B")
    print(f"  Avg ARPU: ${row['ARPU_USD_mean']:.2f}")
    print(f"  Avg Churn: {row['Churn_pct_mean']:.2f}%")
    print(f"  Avg Take Rate: {row['TakeRate_pct_mean']:.3f}%")

# Segment comparison visualization
fig_segment = px.bar(
    segment_stats.reset_index().sort_values('Revenue_USD_M_sum'),
    x='Revenue_USD_M_sum', y='Segment',
    title='Total Revenue by Segment (Full Year 2025)',
    labels={'Revenue_USD_M_sum': 'Revenue (USD Millions)', 'Segment': 'Segment'},
    orientation='h',
    color='Revenue_USD_M_sum',
    color_continuous_scale='Viridis',
    height=500
)
fig_segment.update_layout(font=dict(size=11), showlegend=False)
write_html(fig_segment, 'fintech_outputs/interactive_segment_revenue.html', include_plotlyjs='cdn')
print("\n[SAVED] fintech_outputs/interactive_segment_revenue.html")

# ============================================================================
# SECTION 8: GENERATE COMPREHENSIVE SUMMARY REPORT
# ============================================================================
"""
Create a detailed summary report with key findings and insights
"""

print("\n[REPORT] Generating Summary Report")
print("-" * 80)

summary_report = f"""
{'='*90}
FINTECH DATASET ANALYSIS - COMPREHENSIVE SUMMARY REPORT
{'='*90}

EXECUTIVE SUMMARY
{'-'*90}
Dataset: Fintech Synthetic (2025)
Analysis Period: January 2025 - December 2025
Companies Analyzed: {df['Company'].nunique()}
Business Segments: {df['Segment'].nunique()}
Total Records: {len(df)}
Data Quality: {((1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100):.1f}% complete

MARKET OVERVIEW
{'-'*90}
Total Annual Revenue: ${df['Revenue_USD_M'].sum():,.2f}M
Total Payment Volume (TPV): ${df['TPV_USD_B'].sum():.2f}B
Peak Active Users: {df['Users_M'].max():.1f}M users
Total New Users Acquired: {df['NewUsers_K'].sum():,.0f}K
Average Churn Rate: {df['Churn_pct'].mean():.2f}%
Average Industry ARPU: ${df['ARPU_USD'].mean():.2f}

TOP PERFORMING COMPANIES
{'-'*90}
"""

# Top companies by revenue
top_companies = df.groupby('Company')['Revenue_USD_M'].sum().nlargest(5)
for rank, (company, revenue) in enumerate(top_companies.items(), 1):
    company_data = df[df['Company'] == company]
    users = company_data['Users_M'].max()
    arpu = company_data['ARPU_USD'].mean()
    summary_report += f"{rank}. {company}\n"
    summary_report += f"   Annual Revenue: ${revenue:,.2f}M\n"
    summary_report += f"   Peak Users: {users:.1f}M\n"
    summary_report += f"   Avg ARPU: ${arpu:.2f}\n\n"

# Segment analysis
summary_report += f"""BUSINESS SEGMENT ANALYSIS
{'-'*90}
"""

for segment in df['Segment'].unique():
    segment_data = df[df['Segment'] == segment]
    revenue = segment_data['Revenue_USD_M'].sum()
    companies = segment_data['Company'].nunique()
    summary_report += f"{segment}\n"
    summary_report += f"  Companies: {companies}\n"
    summary_report += f"  Total Revenue: ${revenue:,.2f}M\n"
    summary_report += f"  Avg ARPU: ${segment_data['ARPU_USD'].mean():.2f}\n"
    summary_report += f"  Avg Churn: {segment_data['Churn_pct'].mean():.2f}%\n\n"

# Key insights
summary_report += f"""KEY INSIGHTS & FINDINGS
{'-'*90}
1. Revenue Trends:
   • Total revenue shows {('growth' if df.groupby('Month')['Revenue_USD_M'].sum().iloc[-1] > df.groupby('Month')['Revenue_USD_M'].sum().iloc[0] else 'decline')} from Jan to Dec
   • Highest monthly revenue: ${df.groupby('Month')['Revenue_USD_M'].sum().max():,.2f}M
   • Lowest monthly revenue: ${df.groupby('Month')['Revenue_USD_M'].sum().min():,.2f}M

2. User Acquisition:
   • Total new users in 2025: {df['NewUsers_K'].sum():,.0f}K
   • Peak active users: {df['Users_M'].max():.1f}M
   • Monthly average new users: {df['NewUsers_K'].mean():,.0f}K

3. Customer Economics:
   • Average Customer Acquisition Cost: ${df['CAC_USD'].mean():.2f}
   • Average ARPU: ${df['ARPU_USD'].mean():.2f}
   • Industry average churn: {df['Churn_pct'].mean():.2f}%

4. Payment Volume:
   • Total TPV (annual): ${df['TPV_USD_B'].sum():.2f}B
   • Average monthly TPV: ${df['TPV_USD_B'].mean():.2f}B
   • Highest TPV: ${df['TPV_USD_B'].max():.2f}B

5. Marketing Efficiency:
   • Total marketing spend: ${df['Marketing_Spend_USD_M'].sum():,.2f}M
   • Average CAC Total per month: ${df['CAC_Total_USD_M'].mean():,.2f}M
   • Marketing ROI: {(df['Revenue_USD_M'].sum() / df['Marketing_Spend_USD_M'].sum()):.2f}x

GENERATED OUTPUTS
{'-'*90}
Static Visualizations (PNG):
  • revenue_by_segment_trend.png - Revenue trends across segments
  • tpv_top_companies_trend.png - TPV comparison for top companies
  • user_acquisition_trend.png - User growth and acquisition momentum
  • churn_heatmap.png - Churn rate patterns
  • arpu_by_segment.png - Unit economics by segment
  • correlation_matrix.png - Feature correlations

Interactive Visualizations (HTML):
  • interactive_revenue_timeseries.html - Time-series revenue breakdown
  • interactive_tpv_by_segment.html - TPV by segment
  • interactive_users_scatter.html - User acquisition analysis
  • interactive_arpu_vs_cac.html - Unit economics
  • interactive_revenue_by_segment.html - Segment distribution
  • interactive_latest_month_ranking.html - Company rankings
  • interactive_churn_trends.html - Churn patterns
  • interactive_marketing_efficiency.html - Marketing ROI
  • interactive_company_rankings.html - Comprehensive rankings
  • interactive_segment_revenue.html - Segment comparison

Data Exports:
  • company_rankings.csv - Detailed company performance metrics
  • summary_report.txt - This comprehensive report

{'='*90}
Analysis completed: Fintech Synthetic Data Analysis Pipeline
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*90}
"""

# Write summary report
with open('fintech_outputs/summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("[SAVED] fintech_outputs/summary_report.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE - ALL OUTPUTS GENERATED")
print("="*80)
print("\nGenerated Files in 'fintech_outputs/' directory:")
print("  • PNG Visualizations: 6 static charts")
print("  • HTML Interactive Plots: 9 interactive visualizations")
print("  • Data Exports: company_rankings.csv")
print("  • Reports: summary_report.txt")
print("\nTo view interactive plots, open the .html files in your web browser.")
print("="*80 + "\n")
