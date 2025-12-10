"""
Extract Indicators for All Four Analysis Groups

This script extracts indicators for:
Group 1: Conflict & Stability - "Do cities make us safer?"
Group 2: Economic Inequality - "Does urbanization reduce or widen the gap?"
Group 3: Environmental Impact - "Are cities greener or dirtier?"
Group 4: Development & Quality of Life - "Does urbanization improve human development?"

The script:
1. Extracts relevant indicators for each group
2. Performs data quality checks for each group
3. Identifies outliers and anomalies
4. Saves separate datasets for each analysis group
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data_cleaned/combined_urbanization_life_quality_2008_2020.csv"
OUTPUT_DIR = "data_cleaned/analysis_groups"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*90)
print("INDICATOR EXTRACTION FOR ALL ANALYSIS GROUPS")
print("Urbanization & Life Quality Analysis (2008-2020)")
print("="*90)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"   ✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   ✓ Countries: {df['Country'].nunique()}")
print(f"   ✓ Years: {df['Year'].min()} - {df['Year'].max()}")

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
# STEP 2: DEFINE ALL INDICATOR GROUPS
# ============================================================================
print("\nSTEP 2: Defining Indicators for All Analysis Groups...")

# GROUP 1: CONFLICT & STABILITY
group1_indicators = [
    # Peace & Conflict Metrics
    'overall score', 'internal peace', 'external peace', 'safety and security',
    'ongoing conflict', 'Political instability', 'Political Terror Scale',
    'perceptions of criminality', 'Violent crime', 'violent demonstrations',
    'intensity of internal conflict', 'internal conflicts fought',
    'deaths from internal conflict', 'external conflicts fought',
    'deaths From external conflict', 'terrorism impact',
    # Militarization
    'militarisation', 'military expenditure (% gdp)', 'armed services personnel rate',
    'weapons imports', 'weapons exports', 'nuclear and heavy weapons', 'Access to small arms',
    # Security & Crime
    'homicide rate', 'police rate', 'incarceration rate', 'refugees and idps',
    'Neighbouring countries relations',
    # Peacekeeping
    'un peacekeeping funding'
]

# GROUP 2: ECONOMIC INEQUALITY
group2_indicators = [
    # Inequality
    'Gini coefficient (2021 prices)',
    # Economic Structure
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'Agriculture, forestry, and fishing, value added (annual % growth)',
    # Resource Management
    'Adjusted savings: natural resources depletion (% of GNI)',
    'Adjusted savings: energy depletion (% of GNI)',
    'Adjusted savings: net forest depletion (% of GNI)'
]

# GROUP 3: ENVIRONMENTAL IMPACT
group3_indicators = [
    # Energy & Emissions
    'co2_emiss_excl_lulucf',
    'ren_energy_cons_perc',
    'Adjusted savings: carbon dioxide damage (% of GNI)',
    # Resource Depletion
    'Adjusted savings: natural resources depletion (% of GNI)',
    'Adjusted savings: energy depletion (% of GNI)',
    'Adjusted savings: net forest depletion (% of GNI)',
    # Clean Technologies
    'clean_fuel_tech_cook_pop',
    'Access to clean fuels and technologies for cooking (% of population)'
]

# GROUP 4: DEVELOPMENT & QUALITY OF LIFE
group4_indicators = [
    # Basic Infrastructure
    'elect_access_pop',
    'Access to electricity (% of population)',
    'clean_fuel_tech_cook_pop',
    'Access to clean fuels and technologies for cooking (% of population)',
    # Overall Peace & Development
    'overall score', 'internal peace', 'safety and security',
    # Economic Development
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'Agriculture, forestry, and fishing, value added (annual % growth)',
    # Inequality
    'Gini coefficient (2021 prices)',
    # Crime & Safety
    'homicide rate', 'Violent crime', 'perceptions of criminality'
]

# Create groups dictionary
analysis_groups = {
    'Group1_Conflict_Stability': {
        'name': 'Conflict & Stability',
        'question': 'Do cities make us safer?',
        'indicators': group1_indicators
    },
    'Group2_Economic_Inequality': {
        'name': 'Economic Inequality',
        'question': 'Does urbanization reduce or widen the gap?',
        'indicators': group2_indicators
    },
    'Group3_Environmental': {
        'name': 'Environmental Impact',
        'question': 'Are cities greener or dirtier?',
        'indicators': group3_indicators
    },
    'Group4_Development_QoL': {
        'name': 'Development & Quality of Life',
        'question': 'Does urbanization improve human development?',
        'indicators': group4_indicators
    }
}

print(f"   ✓ Defined {len(analysis_groups)} analysis groups")

# ============================================================================
# STEP 3: EXTRACT AND ANALYZE EACH GROUP
# ============================================================================
print("\nSTEP 3: Extracting and Analyzing Each Group...")

summary_report = []

for group_id, group_info in analysis_groups.items():
    print(f"\n{'='*90}")
    print(f"{group_info['name'].upper()}")
    print(f"Research Question: {group_info['question']}")
    print(f"{'='*90}")
    
    # Combine all columns for this group
    all_cols = identifying_cols + urbanization_cols + group_info['indicators']
    
    # Check which indicators exist
    existing_cols = [col for col in all_cols if col in df.columns]
    missing_cols = [col for col in all_cols if col not in df.columns]
    
    # Remove duplicates while preserving order
    existing_cols = list(dict.fromkeys(existing_cols))
    
    print(f"\n   Indicators:")
    print(f"   ✓ Total requested: {len(all_cols)}")
    print(f"   ✓ Found in dataset: {len(existing_cols)}")
    
    if missing_cols:
        print(f"   ⚠ Missing: {len(missing_cols)}")
        for col in list(dict.fromkeys(missing_cols))[:5]:
            print(f"      - {col}")
        if len(missing_cols) > 5:
            print(f"      ... and {len(missing_cols) - 5} more")
    
    # Extract the subset
    group_df = df[existing_cols].copy()
    
    # Data Quality Check
    print(f"\n   Data Quality:")
    missing_summary = group_df.isnull().sum()
    cols_with_missing = missing_summary[missing_summary > 0]
    
    if len(cols_with_missing) > 0:
        print(f"   ⚠ {len(cols_with_missing)} columns have missing values:")
        for col, count in cols_with_missing.head(5).items():
            pct = (count / len(group_df) * 100)
            print(f"      - {col}: {count} ({pct:.2f}%)")
        if len(cols_with_missing) > 5:
            print(f"      ... and {len(cols_with_missing) - 5} more columns")
    else:
        print(f"   ✓ No missing values!")
    
    # Display all indicators
    print(f"\n   All Indicators in this group:")
    identifying = [col for col in existing_cols if col in identifying_cols]
    urbanization = [col for col in existing_cols if col in urbanization_cols]
    specific = [col for col in existing_cols if col not in identifying_cols and col not in urbanization_cols]
    
    if urbanization:
        print(f"\n   Urbanization Metrics ({len(urbanization)}):")
        for col in urbanization:
            print(f"      • {col}")
    
    if specific:
        print(f"\n   {group_info['name']} Indicators ({len(specific)}):")
        for col in specific:
            print(f"      • {col}")
    
    # Store summary info
    summary_report.append({
        'Group': group_info['name'],
        'Group_ID': group_id,
        'Question': group_info['question'],
        'Total_Indicators': len(all_cols) - len(identifying_cols) - len(urbanization_cols),
        'Found_Indicators': len(existing_cols) - len(identifying_cols) - len(urbanization_cols),
        'Indicator_List': [col for col in existing_cols if col not in identifying_cols],
        'Rows': group_df.shape[0],
        'Columns': group_df.shape[1],
        'Missing_Value_Cols': len(cols_with_missing)
    })

# ============================================================================
# STEP 4: CREATE SUMMARY REPORT WITH INDICATOR LISTS
# ============================================================================
print(f"\n{'='*90}")
print("SUMMARY REPORT - ALL GROUPS")
print(f"{'='*90}\n")

# Save indicators list to text file
indicators_file = os.path.join(OUTPUT_DIR, "ALL_GROUP_INDICATORS.txt")
with open(indicators_file, 'w', encoding='utf-8') as f:
    f.write("="*90 + "\n")
    f.write("INDICATOR LISTS FOR ALL ANALYSIS GROUPS\n")
    f.write("Urbanization & Life Quality Analysis (2008-2020)\n")
    f.write("="*90 + "\n\n")
    
    f.write(f"Dataset: {INPUT_FILE}\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"Countries: {df['Country'].nunique()}\n")
    f.write(f"Years: {df['Year'].min()} - {df['Year'].max()}\n\n")
    
    for info in summary_report:
        f.write("="*90 + "\n")
        f.write(f"{info['Group'].upper()}\n")
        f.write("="*90 + "\n")
        f.write(f"Research Question: {info['Question']}\n")
        f.write(f"Total Indicators: {info['Found_Indicators']}\n\n")
        
        f.write("INDICATORS:\n")
        f.write("-"*90 + "\n\n")
        
        # Separate identifying, urbanization, and specific indicators
        identifying = [col for col in info['Indicator_List'] if col in identifying_cols]
        urbanization = [col for col in info['Indicator_List'] if col in urbanization_cols]
        specific = [col for col in info['Indicator_List'] if col not in identifying_cols and col not in urbanization_cols]
        
        if identifying:
            f.write("Identifying Columns:\n")
            for i, col in enumerate(identifying, 1):
                f.write(f"  {i}. {col}\n")
            f.write("\n")
        
        if urbanization:
            f.write("Urbanization Metrics (Independent Variables):\n")
            for i, col in enumerate(urbanization, 1):
                f.write(f"  {i}. {col}\n")
            f.write("\n")
        
        if specific:
            f.write(f"{info['Group']} Specific Indicators:\n")
            for i, col in enumerate(specific, 1):
                f.write(f"  {i}. {col}\n")
            f.write("\n")
        
        f.write(f"Data Quality:\n")
        f.write(f"  - Missing Values: {info['Missing_Value_Cols']} columns\n")
        f.write("\n\n")

print(f"✓ Indicator lists saved: {indicators_file}")

# Also create a simple summary table
summary_table_file = os.path.join(OUTPUT_DIR, "GROUPS_SUMMARY_TABLE.txt")
with open(summary_table_file, 'w', encoding='utf-8') as f:
    f.write("="*90 + "\n")
    f.write("SUMMARY TABLE - ALL ANALYSIS GROUPS\n")
    f.write("="*90 + "\n\n")
    
    f.write(f"{'Group':<30} {'Question':<45} {'Indicators':>10}\n")
    f.write("-"*90 + "\n")
    
    for info in summary_report:
        f.write(f"{info['Group']:<30} {info['Question'][:44]:<45} {info['Found_Indicators']:>10}\n")
    
    f.write("\n" + "="*90 + "\n")

print(f"✓ Summary table saved: {summary_table_file}")

# ============================================================================
# STEP 5: RECOMMENDATIONS
# ============================================================================
print(f"\n{'='*90}")
print("RECOMMENDED PRE-EDA STEPS FOR EACH GROUP")
print(f"{'='*90}\n")

print("📋 Next Steps for Analysis:\n")
print("1. ✓ Indicator Identification - COMPLETED\n")

print("Recommended Analysis Steps:\n")
print("2. ⚡ Correlation Analysis:")
print("   - Calculate correlations between urbanization and each indicator")
print("   - Create correlation heatmaps for each group")
print("   - Identify strong positive/negative relationships\n")

print("3. ⚡ Temporal Trends:")
print("   - Analyze how indicators changed over time (2008-2020)")
print("   - Compare trends across countries\n")

print("4. ⚡ Cluster Comparison:")
print("   - Compare stable vs volatile clusters")
print("   - Test if urbanization effects differ by cluster\n")

print("5. ⚡ Visualizations:")
print("   - Scatter plots: urbanization vs key indicators")
print("   - Time series plots by country/cluster")
print("   - Box plots for distribution analysis\n")

print(f"\n{'='*90}")
print("EXTRACTION COMPLETE - READY FOR ANALYSIS!")
print(f"{'='*90}")
print(f"\n📁 Files saved:")
print(f"   1. {indicators_file}")
print(f"   2. {summary_table_file}")
print("\n💡 You can now:")
print("   • Use the indicator lists for correlation analysis")
print("   • Perform EDA on each group's indicators")
print("   • Create visualizations based on research questions")
print("   • Compare patterns across clusters")
print(f"\n{'='*90}\n")
