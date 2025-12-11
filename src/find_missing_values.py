"""
Script to identify missing indicator values for recoverable countries
that were excluded from the cleaned dataset.

This script checks if the 8 recoverable countries exist in the cleaned data
and identifies what data is missing for them across all indicators.
"""

import pandas as pd
import os

# Define the 8 recoverable countries
RECOVERABLE_COUNTRIES = {
    'United States': {'missing_years': 1, 'code': 'USA'},
    'Russia': {'missing_years': 1, 'code': 'RUS'},
    'Indonesia': {'missing_years': 1, 'code': 'IDN'},
    'Iran': {'missing_years': 2, 'code': 'IRN'},
    'Kazakhstan': {'missing_years': 1, 'code': 'KAZ'},
    'Bolivia': {'missing_years': 1, 'code': 'BOL'},
    'Uruguay': {'missing_years': 1, 'code': 'URY'},
    'Panama': {'missing_years': 1, 'code': 'PAN'}
}

# Data files
DATA_DIR = 'data_cleaned'
RAW_DATA_DIR = 'data_raw'

cleaned_files = {
    'GPI': os.path.join(DATA_DIR, 'GPI_2008_2020_common_countries.csv'),
    'Gini': os.path.join(DATA_DIR, 'economic_inequality_gini_2008_2020.csv'),
    'Urbanization': os.path.join(DATA_DIR, 'global_urbanization_2008_2020.csv'),
    'HDI': os.path.join(DATA_DIR, 'worlddata_hdi_2008_2020_final.csv')
}

# Expected years
EXPECTED_YEARS = list(range(2008, 2021))  # 2008 to 2020 inclusive

def check_country_in_cleaned_data():
    """Check if recoverable countries exist in cleaned datasets"""
    
    print("="*80)
    print("CHECKING IF RECOVERABLE COUNTRIES EXIST IN CLEANED DATA")
    print("="*80)
    
    results = {}
    
    for dataset_name, file_path in cleaned_files.items():
        print(f"\n### Checking {dataset_name} dataset ###")
        
        try:
            df = pd.read_csv(file_path)
            
            # Get country column name (might vary)
            country_col = None
            for col in df.columns:
                if 'country' in col.lower() or 'entity' in col.lower():
                    country_col = col
                    break
            
            if country_col is None:
                print(f"  ⚠️  Could not find country column in {dataset_name}")
                continue
            
            # Get unique countries in this dataset
            unique_countries = df[country_col].unique()
            
            print(f"  Total countries in {dataset_name}: {len(unique_countries)}")
            
            # Check each recoverable country
            for country_name in RECOVERABLE_COUNTRIES.keys():
                if country_name in unique_countries:
                    print(f"  ✓ {country_name} EXISTS in {dataset_name}")
                    
                    # Get data for this country
                    country_data = df[df[country_col] == country_name]
                    
                    # Check years
                    if 'Year' in df.columns or 'year' in df.columns:
                        year_col = 'Year' if 'Year' in df.columns else 'year'
                        years_present = sorted(country_data[year_col].unique())
                        missing_years = [y for y in EXPECTED_YEARS if y not in years_present]
                        
                        print(f"    Years present: {len(years_present)}/13")
                        if missing_years:
                            print(f"    Missing years: {missing_years}")
                    
                    # Store result
                    if country_name not in results:
                        results[country_name] = {}
                    results[country_name][dataset_name] = 'EXISTS'
                else:
                    print(f"  ✗ {country_name} NOT FOUND in {dataset_name}")
                    
                    # Store result
                    if country_name not in results:
                        results[country_name] = {}
                    results[country_name][dataset_name] = 'NOT FOUND'
        
        except Exception as e:
            print(f"  ❌ Error reading {dataset_name}: {str(e)}")
    
    return results

def check_raw_data_availability():
    """Check if recoverable countries exist in raw data files"""
    
    print("\n" + "="*80)
    print("CHECKING IF RECOVERABLE COUNTRIES EXIST IN RAW DATA")
    print("="*80)
    
    raw_files = {
        'GPI_Combined': os.path.join(RAW_DATA_DIR, 'GPI_2008_2024_combined.csv'),
        'Gini_Raw': os.path.join(RAW_DATA_DIR, 'economic-inequality-gini-index.csv'),
        'Urbanization_Raw': os.path.join(RAW_DATA_DIR, 'global_urbanization_climate_metrics.csv'),
        'WDI_Raw': os.path.join(RAW_DATA_DIR, 'databank_world-development-indicators.csv')
    }
    
    for dataset_name, file_path in raw_files.items():
        if not os.path.exists(file_path):
            print(f"\n⚠️  {dataset_name} file not found: {file_path}")
            continue
            
        print(f"\n### Checking {dataset_name} ###")
        
        try:
            # Read first few rows to understand structure
            df = pd.read_csv(file_path, nrows=5)
            print(f"  Columns: {list(df.columns)[:10]}")  # Show first 10 columns
            
            # Now read full file
            df = pd.read_csv(file_path)
            
            # Find country column
            country_col = None
            for col in df.columns:
                if 'country' in col.lower() or 'entity' in col.lower():
                    country_col = col
                    break
            
            if country_col:
                unique_countries = df[country_col].unique()
                print(f"  Total countries: {len(unique_countries)}")
                
                # Check for recoverable countries
                for country_name in RECOVERABLE_COUNTRIES.keys():
                    if country_name in unique_countries:
                        print(f"  ✓ {country_name} found in raw data")
                    else:
                        # Try with country code
                        code_col = None
                        for col in df.columns:
                            if 'code' in col.lower():
                                code_col = col
                                break
                        
                        if code_col:
                            country_code = RECOVERABLE_COUNTRIES[country_name]['code']
                            if country_code in df[code_col].unique():
                                print(f"  ✓ {country_name} found by code ({country_code})")
                            else:
                                print(f"  ✗ {country_name} not found")
                        else:
                            print(f"  ✗ {country_name} not found")
            else:
                print(f"  ⚠️  Could not identify country column")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

def generate_missing_data_report(results):
    """Generate a comprehensive report of missing data"""
    
    print("\n" + "="*80)
    print("SUMMARY REPORT: MISSING DATA FOR RECOVERABLE COUNTRIES")
    print("="*80)
    
    for country_name, datasets in results.items():
        print(f"\n### {country_name} ###")
        print(f"Expected missing years: {RECOVERABLE_COUNTRIES[country_name]['missing_years']}")
        print(f"Country code: {RECOVERABLE_COUNTRIES[country_name]['code']}")
        
        not_found = [ds for ds, status in datasets.items() if status == 'NOT FOUND']
        exists = [ds for ds, status in datasets.items() if status == 'EXISTS']
        
        if not_found:
            print(f"  ❌ NOT in cleaned data: {', '.join(not_found)}")
        if exists:
            print(f"  ✓ EXISTS in cleaned data: {', '.join(exists)}")
        
        if not not_found:
            print(f"  ℹ️  This country is ALREADY in all cleaned datasets!")
            print(f"     Need to check which specific years/indicators are missing.")

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("MISSING VALUES ANALYSIS FOR RECOVERABLE COUNTRIES")
    print("Data Visualization Project: Life Quality with Increasing Urban Population")
    print("="*80)
    
    print("\nRecoverable Countries to Analyze:")
    for country, info in RECOVERABLE_COUNTRIES.items():
        print(f"  • {country} ({info['code']}) - Missing {info['missing_years']} year(s)")
    
    # Check cleaned data
    results = check_country_in_cleaned_data()
    
    # Check raw data
    check_raw_data_availability()
    
    # Generate report
    generate_missing_data_report(results)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. If countries are ALREADY in cleaned data:
   - They were likely included but may have some missing indicator values
   - Need to identify which specific indicators/years are missing
   
2. If countries are NOT in cleaned data but ARE in raw data:
   - They were filtered out during preprocessing
   - Need to extract their data from raw files
   - Need to identify which years/indicators are missing
   
3. For missing values:
   - Use Perplexity or other sources to find authentic data
   - Ensure data comes from reliable sources (World Bank, UN, etc.)
   - Document the source for each filled value
    """)

if __name__ == "__main__":
    main()
