"""
Extract Indicators for All Four Analysis Groups

This script extracts indicators for:
Group 1: Conflict & Stability - "Do cities make us safer?"
Group 2: Economic Inequality - "Does urbanization reduce or widen the gap?"
Group 3: Environmental Impact - "Are cities greener or dirtier?"
Group 4: Development & Quality of Life - "Does urbanization improve human development?"

The script:
1. Extracts relevant indicators for each group
2. Performs data quality checks
3. Identifies outliers and anomalies
4. Prepares data for EDA and correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_FILE = "data_cleaned/combined_urbanization_life_quality_2008_2020.csv"
OUTPUT_DIR = "data_cleaned/analysis_groups"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("INDICATOR EXTRACTION FOR ALL ANALYSIS GROUPS")
print("Urbanization & Life Quality Analysis (2008-2020)")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"   ✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   ✓ Countries: {df['Country'].nunique()}")
print(f"   ✓ Years: {df['Year'].min()} - {df['Year'].max()}")

# ============================================================================
# STEP 2: DEFINE ALL INDICATOR GROUPS
# ============================================================================
print("\nSTEP 2: Defining Indicators for All Analysis Groups...")

# Core identifying columns (used in all groups)
identifying_cols = ['Country', 'Country_Code', 'Year']

# Urbanization metrics (independent variable - used in all groups)
urbanization_cols = [
    'urban_pop_perc',
    'rural_pop_perc',
    'total_pop',
    'pop_dens_sq_km'
]

# ============================================================================
# GROUP 1: CONFLICT & STABILITY - "Do cities make us safer?"
# ============================================================================
group1_conflict_stability = {
    'Peace_Conflict': [
        'overall score',  # GPI overall score
        'internal peace',
        'external peace',
        'safety and security',
        'ongoing conflict',
        'Political instability',
        'Political Terror Scale',
        'perceptions of criminality',
        'Violent crime',
        'violent demonstrations',
        'intensity of internal conflict',
        'internal conflicts fought',
        'deaths from internal conflict',
        'external conflicts fought',
        'deaths From external conflict',
        'terrorism impact'
    ],
    'Militarization': [
        'militarisation',
        'military expenditure (% gdp)',
        'armed services personnel rate',
        'weapons imports',
        'weapons exports',
        'nuclear and heavy weapons',
        'Access to small arms'
    ],
    'Security_Crime': [
        'homicide rate',
        'police rate',
        'incarceration rate',
        'refugees and idps',
        'Neighbouring countries relations'
    ],
    'Peacekeeping': [
        'un peacekeeping funding'
    ]
}

# ============================================================================
# GROUP 2: ECONOMIC INEQUALITY - "Does urbanization reduce or widen the gap?"
# ============================================================================
group2_economic_inequality = {
    'Inequality': [
        'Gini coefficient (2021 prices)'
    ],
    'Economic_Structure': [
        'Agriculture, forestry, and fishing, value added (% of GDP)',
        'Agriculture, forestry, and fishing, value added (annual % growth)'
    ],
    'Resource_Management': [
        'Adjusted savings: natural resources depletion (% of GNI)',
        'Adjusted savings: energy depletion (% of GNI)',
        'Adjusted savings: net forest depletion (% of GNI)'
    ]
}

# ============================================================================
# GROUP 3: ENVIRONMENTAL IMPACT - "Are cities greener or dirtier?"
# ============================================================================
group3_environmental = {
    'Energy_Emissions': [
        'co2_emiss_excl_lulucf',
        'ren_energy_cons_perc',
        'Adjusted savings: carbon dioxide damage (% of GNI)'
    ],
    'Resource_Depletion': [
        'Adjusted savings: natural resources depletion (% of GNI)',
        'Adjusted savings: energy depletion (% of GNI)',
        'Adjusted savings: net forest depletion (% of GNI)'
    ],
    'Clean_Technologies': [
        'clean_fuel_tech_cook_pop',
        'Access to clean fuels and technologies for cooking (% of population)'
    ]
}

# ============================================================================
# GROUP 4: DEVELOPMENT & QUALITY OF LIFE - "Does urbanization improve human development?"
# ============================================================================
group4_development = {
    'Basic_Infrastructure': [
        'elect_access_pop',
        'Access to electricity (% of population)',
        'clean_fuel_tech_cook_pop',
        'Access to clean fuels and technologies for cooking (% of population)'
    ],
    'Overall_Peace_Development': [
        'overall score',
        'internal peace',
        'safety and security'
    ],
    'Economic_Development': [
        'Agriculture, forestry, and fishing, value added (% of GDP)',
        'Agriculture, forestry, and fishing, value added (annual % growth)'
    ],
    'Inequality': [
        'Gini coefficient (2021 prices)'
    ],
    'Crime_Safety': [
        'homicide rate',
        'Violent crime',
        'perceptions of criminality'
    ]
}

# Combine all groups
all_groups = {
    'Group1_Conflict_Stability': group1_conflict_stability,
    'Group2_Economic_Inequality': group2_economic_inequality,
    'Group3_Environmental': group3_environmental,
    'Group4_Development_QoL': group4_development
}

print(f"   ✓ Defined 4 analysis groups")

# ============================================================================
# STEP 3: DATA QUALITY ASSESSMENT
# ============================================================================
print("\nSTEP 3: Data Quality Assessment...")

# Missing values analysis
print("\n   A. Missing Values Analysis:")
missing_summary = conflict_df.isnull().sum()
missing_pct = (missing_summary / len(conflict_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing_summary.index,
    'Missing_Count': missing_summary.values,
    'Missing_Percent': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(f"      ⚠ {len(missing_df)} columns have missing values:")
    for _, row in missing_df.head(10).iterrows():
        print(f"         - {row['Column']}: {row['Missing_Count']} ({row['Missing_Percent']}%)")
else:
    print("      ✓ No missing values found!")

# Data types
print("\n   B. Data Types:")
numeric_cols = conflict_df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [col for col in existing_indicators if col not in numeric_cols]
print(f"      - Numeric columns: {len(numeric_cols)}")
print(f"      - Non-numeric columns: {len(non_numeric_cols)} {non_numeric_cols}")

# ============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ============================================================================
print("\nSTEP 4: Generating Descriptive Statistics...")

# Get numeric columns only (excluding identifiers)
analysis_cols = [col for col in numeric_cols if col not in ['Country_Code', 'Year']]

# Calculate statistics
stats_df = conflict_df[analysis_cols].describe().T
stats_df['range'] = stats_df['max'] - stats_df['min']
stats_df['cv'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)  # Coefficient of variation

print(f"   ✓ Statistics calculated for {len(analysis_cols)} indicators")
print("\n   Sample statistics (first 5 indicators):")
print(stats_df.head()[['mean', 'std', 'min', 'max', 'cv']])

# ============================================================================
# STEP 5: OUTLIER DETECTION
# ============================================================================
print("\nSTEP 5: Outlier Detection (IQR Method)...")

outlier_summary = []
for col in analysis_cols:
    if col in conflict_df.columns:
        Q1 = conflict_df[col].quantile(0.25)
        Q3 = conflict_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = conflict_df[(conflict_df[col] < lower_bound) | (conflict_df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(conflict_df) * 100).round(2)
        
        if outlier_count > 0:
            outlier_summary.append({
                'Indicator': col,
                'Outlier_Count': outlier_count,
                'Outlier_Percent': outlier_pct,
                'Lower_Bound': round(lower_bound, 2),
                'Upper_Bound': round(upper_bound, 2)
            })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Count', ascending=False)
print(f"   ✓ Indicators with outliers: {len(outlier_df)}")
if len(outlier_df) > 0:
    print("\n   Top indicators with most outliers:")
    for _, row in outlier_df.head(5).iterrows():
        print(f"      - {row['Indicator']}: {row['Outlier_Count']} ({row['Outlier_Percent']}%)")

# ============================================================================
# STEP 6: CORRELATION PREVIEW
# ============================================================================
print("\nSTEP 6: Correlation Preview (Urbanization vs Conflict Indicators)...")

# Calculate correlation between urban_pop_perc and key conflict indicators
key_conflict_indicators = [
    'Political instability',
    'overall score',
    'homicide rate',
    'Violent crime',
    'terrorism impact',
    'military expenditure (% gdp)'
]

existing_key = [col for col in key_conflict_indicators if col in conflict_df.columns]
if 'urban_pop_perc' in conflict_df.columns and existing_key:
    correlations = {}
    for indicator in existing_key:
        corr = conflict_df[['urban_pop_perc', indicator]].corr().iloc[0, 1]
        correlations[indicator] = corr
    
    print("\n   Correlation with Urban Population %:")
    for indicator, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "↑" if corr > 0 else "↓"
        print(f"      {direction} {indicator}: {corr:.3f}")

# ============================================================================
# STEP 7: SAVE OUTPUTS
# ============================================================================
print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Save extracted data
conflict_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Extracted data saved: {OUTPUT_FILE}")
print(f"  Shape: {conflict_df.shape}")

# Save summary report
with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CONFLICT & STABILITY INDICATORS - DATA QUALITY REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. DATASET OVERVIEW\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Records: {len(conflict_df)}\n")
    f.write(f"Countries: {conflict_df['Country'].nunique()}\n")
    f.write(f"Years: {conflict_df['Year'].min()} - {conflict_df['Year'].max()}\n")
    f.write(f"Total Indicators: {len(existing_indicators)}\n\n")
    
    f.write("2. INDICATOR CATEGORIES\n")
    f.write("-" * 40 + "\n")
    for category, indicators in conflict_stability_indicators.items():
        existing = [i for i in indicators if i in df.columns]
        f.write(f"{category}: {len(existing)} indicators\n")
        for ind in existing:
            f.write(f"   - {ind}\n")
        f.write("\n")
    
    f.write("3. MISSING VALUES\n")
    f.write("-" * 40 + "\n")
    if len(missing_df) > 0:
        f.write(missing_df.to_string(index=False))
    else:
        f.write("No missing values found!\n")
    f.write("\n\n")
    
    f.write("4. OUTLIERS (IQR Method)\n")
    f.write("-" * 40 + "\n")
    if len(outlier_df) > 0:
        f.write(outlier_df.to_string(index=False))
    else:
        f.write("No outliers detected\n")
    f.write("\n\n")
    
    f.write("5. DESCRIPTIVE STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(stats_df.to_string())
    f.write("\n")

print(f"✓ Summary report saved: {OUTPUT_SUMMARY}")

# ============================================================================
# STEP 8: RECOMMENDATIONS FOR NEXT STEPS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDED STEPS BEFORE EDA")
print("="*80)

print("\n📋 Pre-EDA Checklist:")
print("\n   1. ✓ Data Extraction - COMPLETED")
print("   2. ✓ Missing Values Assessment - COMPLETED")
print("   3. ✓ Outlier Detection - COMPLETED")
print("   4. ✓ Descriptive Statistics - COMPLETED")
print("\n   Next Steps:")
print("   5. ⚡ Handle Missing Values:")
print("      - Decide: Impute (mean/median) or drop rows/columns")
print("      - Consider forward/backward fill for time series")
print("\n   6. ⚡ Handle Outliers:")
print("      - Investigate outliers (are they errors or genuine extreme values?)")
print("      - Decide: Keep, transform (log/sqrt), or cap (winsorize)")
print("\n   7. ⚡ Check Distributions:")
print("      - Plot histograms/density plots for each indicator")
print("      - Check for skewness and normality")
print("      - Consider transformations for highly skewed data")
print("\n   8. ⚡ Check Multicollinearity:")
print("      - Calculate VIF (Variance Inflation Factor)")
print("      - Remove highly correlated redundant variables")
print("\n   9. ⚡ Temporal Consistency:")
print("      - Check if trends are consistent across years")
print("      - Identify any sudden jumps or data collection issues")
print("\n   10. ⚡ Cluster-wise Analysis:")
print("       - Compare patterns between the two clusters")
print("       - Test if urbanization effects differ by cluster")

print("\n" + "="*80)
print("READY FOR EXPLORATORY DATA ANALYSIS!")
print("="*80)
print(f"\n📊 Dataset ready at: {OUTPUT_FILE}")
print(f"📄 Full report at: {OUTPUT_SUMMARY}")
print("\n💡 You can now:")
print("   • Visualize distributions and trends")
print("   • Calculate correlations between urbanization and conflict indicators")
print("   • Create scatter plots and heatmaps")
print("   • Compare patterns across clusters")
print("   • Test hypotheses about urbanization and safety")
print("\n" + "="*80)
