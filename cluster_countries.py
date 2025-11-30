"""
K-Means Clustering Script for Country Classification
Divides 46 countries into 2 clusters based on urbanization, development, and stability indicators

Cluster Goal:
- Cluster 0: Urbanized/Rich/Stable countries
- Cluster 1: Urbanizing/Developing/Volatile/Unequal countries

This demonstrates: "Urbanization is not a magic bullet; it depends on how you urbanize"
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_FILE = "data_cleaned/combined_urbanization_life_quality_2008_2020.csv"
OUTPUT_FILE = "data_cleaned/Country_Cluster_Key.csv"
DETAILED_OUTPUT = "data_cleaned/Country_Cluster_Profiles.csv"

print("="*70)
print("K-MEANS CLUSTERING: Urbanization & Life Quality Analysis")
print("="*70)
print("\nObjective: Divide 46 countries into 2 meaningful clusters")
print("Hypothesis: Urbanization outcomes depend on HOW you urbanize\n")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"   ✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   ✓ Countries: {df['Country'].nunique()}")
print(f"   ✓ Years: {df['Year'].min()} - {df['Year'].max()}")

# ============================================================================
# STEP 2: CREATE COUNTRY PROFILES (Aggregation)
# ============================================================================
print("\nSTEP 2: Creating Country Profiles (Aggregating 2008-2020 data)...")
print("   → Computing MEAN of each indicator across all years per country")

# Select only numeric columns for clustering (exclude Country, Country_Code, Year)
id_columns = ['Country', 'Country_Code', 'Year']
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Create country profiles by averaging across years
country_profiles = df.groupby('Country')[numeric_columns].mean().reset_index()
print(f"   ✓ Created profiles: {country_profiles.shape[0]} countries × {len(numeric_columns)} indicators")

# ============================================================================
# STEP 3: NORMALIZE DATA (Standardization)
# ============================================================================
print("\nSTEP 3: Normalizing data (StandardScaler)...")
print("   → Converting all indicators to same scale (mean=0, std=1)")
print("   → This ensures equal 'voting power' for all features")

# Extract feature matrix for clustering
X = country_profiles[numeric_columns].values
countries = country_profiles['Country'].values

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   ✓ Scaled features: {X_scaled.shape}")

# ============================================================================
# STEP 4: APPLY K-MEANS CLUSTERING
# ============================================================================
print("\nSTEP 4: Applying K-Means Clustering (k=2)...")
print("   → Finding optimal way to divide countries into 2 groups")

# Set random state for reproducibility
np.random.seed(42)

# Apply K-Means with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to country profiles
country_profiles['Cluster'] = clusters
print(f"   ✓ Clustering complete!")
print(f"   ✓ Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

# ============================================================================
# STEP 5: ANALYZE CLUSTERS
# ============================================================================
print("\nSTEP 5: Analyzing Cluster Characteristics...")

cluster_0_countries = country_profiles[country_profiles['Cluster'] == 0]['Country'].tolist()
cluster_1_countries = country_profiles[country_profiles['Cluster'] == 1]['Country'].tolist()

print(f"\n   Cluster 0: {len(cluster_0_countries)} countries")
print(f"   Cluster 1: {len(cluster_1_countries)} countries")

# Calculate cluster statistics for key indicators
key_indicators = [
    'urban_pop_perc',
    'Gini coefficient (2021 prices)',
    'overall score',  # GPI overall score (lower = more peaceful)
    'total_pop',
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'co2_emiss_excl_lulucf',
    'Political instability',
    'homicide rate'
]

print("\n   Key Indicator Comparison (Mean values):")
print("   " + "-"*66)
print(f"   {'Indicator':<40} {'Cluster 0':<12} {'Cluster 1':<12}")
print("   " + "-"*66)

for indicator in key_indicators:
    if indicator in country_profiles.columns:
        c0_mean = country_profiles[country_profiles['Cluster'] == 0][indicator].mean()
        c1_mean = country_profiles[country_profiles['Cluster'] == 1][indicator].mean()
        print(f"   {indicator:<40} {c0_mean:>11.2f} {c1_mean:>11.2f}")

# ============================================================================
# STEP 6: INTERPRET CLUSTERS
# ============================================================================
print("\n" + "="*70)
print("CLUSTER INTERPRETATION")
print("="*70)

# Determine which cluster is "developed" vs "developing" based on key metrics
c0_gpi = country_profiles[country_profiles['Cluster'] == 0]['overall score'].mean()
c1_gpi = country_profiles[country_profiles['Cluster'] == 1]['overall score'].mean()
c0_gini = country_profiles[country_profiles['Cluster'] == 0]['Gini coefficient (2021 prices)'].mean()
c1_gini = country_profiles[country_profiles['Cluster'] == 1]['Gini coefficient (2021 prices)'].mean()

# Lower GPI score = more peaceful, Lower Gini = more equal
if c0_gpi < c1_gpi and c0_gini < c1_gini:
    stable_cluster = 0
    volatile_cluster = 1
    stable_label = "Urbanized/Rich/Stable"
    volatile_label = "Urbanizing/Developing/Volatile"
else:
    stable_cluster = 1
    volatile_cluster = 0
    stable_label = "Urbanized/Rich/Stable"
    volatile_label = "Urbanizing/Developing/Volatile"

print(f"\nCluster {stable_cluster}: {stable_label}")
print(f"   → Lower conflict, better peace scores")
print(f"   → Lower inequality (Gini coefficient)")
print(f"   → Higher stability")
print(f"   Countries ({len(country_profiles[country_profiles['Cluster'] == stable_cluster])}):")
stable_countries = country_profiles[country_profiles['Cluster'] == stable_cluster]['Country'].tolist()
for i in range(0, len(stable_countries), 5):
    print(f"      {', '.join(stable_countries[i:i+5])}")

print(f"\nCluster {volatile_cluster}: {volatile_label}")
print(f"   → Higher conflict, volatility")
print(f"   → Higher inequality")
print(f"   → More developmental challenges")
print(f"   Countries ({len(country_profiles[country_profiles['Cluster'] == volatile_cluster])}):")
volatile_countries = country_profiles[country_profiles['Cluster'] == volatile_cluster]['Country'].tolist()
for i in range(0, len(volatile_countries), 5):
    print(f"      {', '.join(volatile_countries[i:i+5])}")

# ============================================================================
# STEP 7: SAVE OUTPUTS
# ============================================================================
print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# Save simple cluster key (Country + Cluster ID)
cluster_key = country_profiles[['Country', 'Cluster']].copy()
cluster_key = cluster_key.sort_values('Country')
cluster_key.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Cluster Key saved: {OUTPUT_FILE}")
print(f"  Format: Country, Cluster (0 or 1)")

# Save detailed profiles with all indicators
country_profiles_sorted = country_profiles.sort_values(['Cluster', 'Country'])
country_profiles_sorted.to_csv(DETAILED_OUTPUT, index=False)
print(f"✓ Detailed Profiles saved: {DETAILED_OUTPUT}")
print(f"  Format: Country, All Indicators, Cluster")

# ============================================================================
# STEP 8: VISUALIZE CLUSTERS (Optional - PCA for 2D visualization)
# ============================================================================
print("\nSTEP 8: Creating visualization...")

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create visualization
plt.figure(figsize=(12, 8))
colors = ['#2E86AB', '#A23B72']
labels = [stable_label, volatile_label]

for cluster_id in [0, 1]:
    mask = clusters == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[cluster_id], label=f'Cluster {cluster_id}: {labels[cluster_id]}',
                s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

# Add country labels to points
for i, country in enumerate(countries):
    plt.annotate(country, (X_pca[i, 0], X_pca[i, 1]), 
                fontsize=7, alpha=0.7, ha='center')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('K-Means Clustering: Urbanization & Life Quality (2008-2020)\n46 Countries, 2 Clusters', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plot_file = "data_cleaned/Country_Clusters_Visualization.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved: {plot_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("CLUSTERING COMPLETE!")
print("="*70)
print(f"\n📊 Results Summary:")
print(f"   • Total countries clustered: {len(countries)}")
print(f"   • Cluster 0 ({stable_label}): {len(cluster_0_countries)} countries")
print(f"   • Cluster 1 ({volatile_label}): {len(cluster_1_countries)} countries")
print(f"   • Features used: {len(numeric_columns)} indicators")
print(f"   • PCA variance explained (2 components): {sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"\n📁 Output Files:")
print(f"   1. {OUTPUT_FILE}")
print(f"   2. {DETAILED_OUTPUT}")
print(f"   3. {plot_file}")
print(f"\n💡 Key Insight:")
print(f"   'Urbanization is not a magic bullet; it depends on HOW you urbanize'")
print(f"   The clusters show distinct patterns in how urbanization correlates with")
print(f"   development, stability, and quality of life across different contexts.")
print("\n" + "="*70)
