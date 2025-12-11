"""
Conflict & Stability: Anomaly and Outlier Detection
Theme: Do cities make us safer?

This script performs outlier detection on specific conflict and stability indicators
using the Interquartile Range (IQR) method to identify anomalies across countries and years.
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data_cleaned/combined_urbanization_life_quality_2008_2020.csv"
OUTPUT_DIR = "data_cleaned/conflict_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the specific indicators for this group
indicators = [
    "Political instability",
    "Political Terror Scale",
    "internal peace",
    "intensity of internal conflict",
    "deaths from internal conflict",
    "internal conflicts fought",
    "terrorism impact",
    "militarisation",
    "military expenditure (% gdp)",
    "armed services personnel rate",
    "weapons exports",
    "weapons imports",
    "nuclear and heavy weapons",
    "un peacekeeping funding",
    "external conflicts fought",
    "deaths From external conflict",
    "Neighbouring countries relations",
    "external peace",
    "ongoing conflict",
    "refugees and idps",
    "overall score"
]

print("="*80)
print("CONFLICT & STABILITY: ANOMALY DETECTION")
print("="*80)

# Load data
if not os.path.exists(INPUT_FILE):
    print(f"Error: Input file not found at {INPUT_FILE}")
    exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# Check for missing indicators
existing_indicators = [col for col in indicators if col in df.columns]
missing_indicators = [col for col in indicators if col not in df.columns]

if missing_indicators:
    print(f"\n⚠ Warning: {len(missing_indicators)} indicators not found in dataset:")
    for col in missing_indicators:
        print(f"  - {col}")

print(f"\nAnalyzing {len(existing_indicators)} indicators for outliers...")

# ---------------------------------------------------------
# Outlier Detection (IQR Method)
# ---------------------------------------------------------
outlier_rows = []
summary_stats = []

for col in existing_indicators:
    # Skip non-numeric columns just in case
    if not np.issubdtype(df[col].dtype, np.number):
        print(f"Skipping non-numeric column: {col}")
        continue
        
    # Calculate IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    # Store summary statistics
    summary_stats.append({
        'Indicator': col,
        'Outlier_Count': len(outliers),
        'Percentage': round(len(outliers) / len(df) * 100, 2),
        'Lower_Bound': round(lower_bound, 4),
        'Upper_Bound': round(upper_bound, 4),
        'Min_Value': round(df[col].min(), 4),
        'Max_Value': round(df[col].max(), 4),
        'Mean': round(df[col].mean(), 4)
    })
    
    # Store detailed outlier rows
    for _, row in outliers.iterrows():
        outlier_rows.append({
            'Country': row['Country'],
            'Year': row['Year'],
            'Indicator': col,
            'Value': row[col],
            'Type': 'Low' if row[col] < lower_bound else 'High',
            'Threshold': lower_bound if row[col] < lower_bound else upper_bound,
            'Deviation': row[col] - (lower_bound if row[col] < lower_bound else upper_bound)
        })

# Create DataFrames
outliers_df = pd.DataFrame(outlier_rows)
summary_df = pd.DataFrame(summary_stats).sort_values('Outlier_Count', ascending=False)

# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------
detailed_file = os.path.join(OUTPUT_DIR, "conflict_outliers_detailed.csv")
summary_file = os.path.join(OUTPUT_DIR, "conflict_outliers_summary.csv")
report_file = os.path.join(OUTPUT_DIR, "conflict_anomalies_report.txt")

outliers_df.to_csv(detailed_file, index=False)
summary_df.to_csv(summary_file, index=False)

# Generate Text Report
with open(report_file, "w", encoding="utf-8") as f:
    f.write("CONFLICT & STABILITY: ANOMALY DETECTION REPORT\n")
    f.write("==============================================\n")
    f.write(f"Total Records Analyzed: {len(df)}\n")
    f.write(f"Indicators Analyzed: {len(existing_indicators)}\n")
    f.write(f"Total Anomalies Detected: {len(outliers_df)}\n\n")
    
    f.write("1. INDICATORS WITH HIGHEST ANOMALY RATES\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Indicator':<35} {'Count':<10} {'Percent':<10}\n")
    f.write("-" * 50 + "\n")
    for _, row in summary_df.head(10).iterrows():
        f.write(f"{row['Indicator']:<35} {row['Outlier_Count']:<10} {row['Percentage']}% \n")
    f.write("\n")
    
    f.write("2. COUNTRIES WITH MOST ANOMALIES (Cumulative across all indicators)\n")
    f.write("-" * 50 + "\n")
    if not outliers_df.empty:
        top_countries = outliers_df['Country'].value_counts().head(10)
        for country, count in top_countries.items():
            f.write(f"{country:<35} {count} anomalies\n")
    else:
        f.write("No anomalies detected.\n")
    f.write("\n")
    
    f.write("3. EXTREME ANOMALIES (Top 5 Highest Values per Indicator)\n")
    f.write("-" * 50 + "\n")
    for col in existing_indicators:
        if not np.issubdtype(df[col].dtype, np.number): continue
        
        f.write(f"\nIndicator: {col}\n")
        # Get top 5 highest values
        top_vals = df.nlargest(5, col)[['Country', 'Year', col]]
        for _, row in top_vals.iterrows():
            f.write(f"  • {row['Country']} ({row['Year']}): {row[col]}\n")

print(f"\nAnalysis Complete!")
print(f"✓ Detailed outliers saved to: {detailed_file}")
print(f"✓ Summary stats saved to: {summary_file}")
print(f"✓ Full report saved to: {report_file}")
