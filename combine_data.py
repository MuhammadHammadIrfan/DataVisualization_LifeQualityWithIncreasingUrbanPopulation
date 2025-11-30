"""
Script to combine all cleaned data files into one comprehensive dataset
for analyzing the effect of urbanization on life quality (2008-2020)
"""

import pandas as pd
import os

# Define the data directory
data_dir = "data_cleaned"

# Read all four cleaned data files
print("Reading data files...")

# 1. Economic Inequality (Gini coefficient)
gini_df = pd.read_csv(os.path.join(data_dir, "economic_inequality_gini_2008_2020.csv"))
print(f"Gini data: {gini_df.shape}")

# 2. Global Urbanization
urbanization_df = pd.read_csv(os.path.join(data_dir, "global_urbanization_2008_2020.csv"))
print(f"Urbanization data: {urbanization_df.shape}")

# 3. Global Peace Index (GPI)
gpi_df = pd.read_csv(os.path.join(data_dir, "GPI_2008_2020_common_countries.csv"))
print(f"GPI data: {gpi_df.shape}")

# 4. Human Development Index (HDI) and related indicators
hdi_df = pd.read_csv(os.path.join(data_dir, "worlddata_hdi_2008_2020_final.csv"))
print(f"HDI data: {hdi_df.shape}")

# Standardize column names for merging
print("\nStandardizing column names...")

# Rename columns to have consistent names for merging
gini_df = gini_df.rename(columns={
    'Entity': 'Country',
    'Code': 'Country_Code'
})

urbanization_df = urbanization_df.rename(columns={
    'country': 'Country',
    'country_code': 'Country_Code',
    'year': 'Year'
})

gpi_df = gpi_df.rename(columns={
    'country': 'Country',
    'geocode': 'Country_Code'
})

hdi_df = hdi_df.rename(columns={
    'Country Name': 'Country',
    'Country Code': 'Country_Code'
})

# Start merging datasets
print("\nMerging datasets...")

# Start with urbanization data as the base (most comprehensive)
combined_df = urbanization_df.copy()

# Merge with Gini coefficient data
combined_df = combined_df.merge(
    gini_df[['Country', 'Country_Code', 'Year', 'Gini coefficient (2021 prices)']],
    on=['Country', 'Country_Code', 'Year'],
    how='left'
)
print(f"After merging Gini: {combined_df.shape}")

# Merge with GPI data
# GPI has many columns, so we'll select key indicators
gpi_columns = ['Country', 'Country_Code', 'Year'] + [col for col in gpi_df.columns 
                                                       if col not in ['Country', 'Country_Code', 'Year']]
combined_df = combined_df.merge(
    gpi_df[gpi_columns],
    on=['Country', 'Country_Code', 'Year'],
    how='left'
)
print(f"After merging GPI: {combined_df.shape}")

# Merge with HDI data
# HDI also has many columns
hdi_columns = ['Country', 'Country_Code', 'Year'] + [col for col in hdi_df.columns 
                                                      if col not in ['Country', 'Country_Code', 'Year']]
combined_df = combined_df.merge(
    hdi_df[hdi_columns],
    on=['Country', 'Country_Code', 'Year'],
    how='left',
    suffixes=('', '_hdi')
)
print(f"After merging HDI: {combined_df.shape}")

# Reorder columns for better readability
# Put identifying columns first
id_cols = ['Country', 'Country_Code', 'Year']
other_cols = [col for col in combined_df.columns if col not in id_cols]
combined_df = combined_df[id_cols + other_cols]

# Sort by country and year
combined_df = combined_df.sort_values(['Country', 'Year']).reset_index(drop=True)

# Save the combined dataset
output_path = os.path.join(data_dir, "combined_urbanization_life_quality_2008_2020.csv")
combined_df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"Combined dataset created successfully!")
print(f"{'='*60}")
print(f"Output file: {output_path}")
print(f"Shape: {combined_df.shape}")
print(f"Countries: {combined_df['Country'].nunique()}")
print(f"Years covered: {combined_df['Year'].min()} - {combined_df['Year'].max()}")
print(f"\nColumn categories:")
print(f"  - Identifying columns: 3 (Country, Country_Code, Year)")
print(f"  - Urbanization indicators: {len([col for col in combined_df.columns if col in urbanization_df.columns]) - 3}")
print(f"  - Economic inequality: 1 (Gini coefficient)")
print(f"  - Peace indicators: {len([col for col in combined_df.columns if col in gpi_df.columns]) - 3}")
print(f"  - Development indicators: {len([col for col in combined_df.columns if col in hdi_df.columns]) - 3}")
print(f"  - Total indicators: {len(combined_df.columns) - 3}")

# Display first few rows
print(f"\nFirst few rows of combined data:")
print(combined_df.head())

# Check for missing values
print(f"\n{'='*60}")
print(f"Missing values summary:")
print(f"{'='*60}")
missing_summary = combined_df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("No missing values found!")

print(f"\nData combination complete!")
