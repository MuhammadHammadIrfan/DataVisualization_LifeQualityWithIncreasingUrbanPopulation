"""
Interactive Urbanization Analysis Dashboard
============================================
Two linked visualizations:
1. Multi-indicator comparison across urbanization levels
2. Deep-dive scatter plot analysis with clustering comparison

WCAG/508 Compliant Color Palette
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = "../data_cleaned/combined_urbanization_life_quality_2008_2020.csv"

# WCAG/508 Compliant Colors (from accessibility guidelines)
COLORS = {
    'urbanization_low': '#377EB8',      # Blue (Low Urbanization)
    'urbanization_medium': '#E41A1C',   # Orange-Red (Medium Urbanization)
    'urbanization_high': '#4DAF4A',     # Green (High Urbanization)
    'stable_cluster': '#377EB8',        # Blue (Stable Urbanizers)
    'volatile_cluster': '#E41A1C',      # Orange-Red (Volatile Urbanizers)
    'all_countries': '#999999',         # Gray (All Countries Combined)
}

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"✓ Loaded: {df.shape[0]} rows, {df['Country'].nunique()} countries")

# ============================================================================
# STEP 2: CREATE COUNTRY CLUSTERS (Stable vs Volatile Urbanizers)
# ============================================================================
print("\nCreating country clusters...")

# Select numeric columns for clustering
id_columns = ['Country', 'Country_Code', 'Year']
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns = [col for col in numeric_columns if col != 'Year']

# Create country profiles by averaging across years
country_profiles = df.groupby('Country')[numeric_columns].mean().reset_index()

# Normalize and cluster
X = country_profiles[numeric_columns].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means (k=2)
np.random.seed(42)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to country profiles
country_profiles['Cluster'] = clusters

# Determine which cluster is Stable vs Volatile based on peace scores
c0_peace = country_profiles[country_profiles['Cluster'] == 0]['overall score'].mean()
c1_peace = country_profiles[country_profiles['Cluster'] == 1]['overall score'].mean()

if c0_peace < c1_peace:  # Lower GPI score = more peaceful
    stable_cluster_id = 0
    volatile_cluster_id = 1
else:
    stable_cluster_id = 1
    volatile_cluster_id = 0

# Create cluster mapping
country_cluster_map = {}
for _, row in country_profiles.iterrows():
    if row['Cluster'] == stable_cluster_id:
        country_cluster_map[row['Country']] = 'Stable Urbanizers'
    else:
        country_cluster_map[row['Country']] = 'Volatile Urbanizers'

# Add cluster column to main dataframe
df['Country_Cluster'] = df['Country'].map(country_cluster_map)

print(f"✓ Cluster 0 (Stable): {sum(clusters == stable_cluster_id)} countries")
print(f"✓ Cluster 1 (Volatile): {sum(clusters == volatile_cluster_id)} countries")

# ============================================================================
# STEP 3: CALCULATE CORRELATIONS WITH URBANIZATION
# ============================================================================
print("\nCalculating correlations...")

urban_metric = 'urban_pop_perc'
conflict_indicators = [col for col in numeric_columns if col != urban_metric]

# Calculate correlations
correlations = {}
for indicator in conflict_indicators:
    if indicator in df.columns:
        corr = df[[urban_metric, indicator]].corr().iloc[0, 1]
        if not np.isnan(corr):
            correlations[indicator] = corr

# Sort by correlation strength
corr_series = pd.Series(correlations).sort_values()

# Select top 5 positive and top 5 negative correlations
top_5_positive = corr_series.tail(5).index.tolist()
top_5_negative = corr_series.head(5).index.tolist()
selected_indicators = top_5_negative + top_5_positive

# Ensure 'overall score' (Global Peace Index) is included
if 'overall score' not in selected_indicators:
    # Replace the weakest correlation with overall score
    if abs(corr_series[top_5_negative[-1]]) < abs(corr_series[top_5_positive[0]]):
        selected_indicators[4] = 'overall score'  # Replace last of bottom 5
    else:
        selected_indicators[5] = 'overall score'  # Replace first of top 5

print(f"✓ Selected {len(selected_indicators)} key indicators")

# ============================================================================
# STEP 4: CREATE URBANIZATION GROUPS
# ============================================================================
print("\nCreating urbanization groups...")

df['Urban_Group'] = pd.cut(
    df[urban_metric], 
    bins=[0, 50, 75, 100], 
    labels=['Low Urbanization (<50%)', 'Medium Urbanization (50-75%)', 'High Urbanization (>75%)'],
    include_lowest=True
)

# Calculate group means for selected indicators
group_means = df.groupby('Urban_Group', observed=True)[selected_indicators].mean()

# Normalize to 0-1 scale for visualization
from sklearn.preprocessing import MinMaxScaler
scaler_viz = MinMaxScaler()
group_means_normalized = pd.DataFrame(
    scaler_viz.fit_transform(group_means),
    index=group_means.index,
    columns=group_means.columns
)

print("✓ Urbanization groups created")

# ============================================================================
# VISUALIZATION 1: MULTI-INDICATOR COMPARISON
# ============================================================================
print("\nCreating Visualization 1: Multi-Indicator Comparison...")

# Prepare data for grouped bar chart
viz1_data = []
for urban_group in group_means_normalized.index:
    for indicator in group_means_normalized.columns:
        viz1_data.append({
            'Indicator': indicator,
            'Urban_Group': urban_group,
            'Normalized_Score': group_means_normalized.loc[urban_group, indicator],
            'Correlation': correlations.get(indicator, 0)
        })

viz1_df = pd.DataFrame(viz1_data)

# Create grouped bar chart
fig1 = go.Figure()

urban_groups = ['Low Urbanization (<50%)', 'Medium Urbanization (50-75%)', 'High Urbanization (>75%)']
colors_urban = [COLORS['urbanization_low'], COLORS['urbanization_medium'], COLORS['urbanization_high']]

for i, urban_group in enumerate(urban_groups):
    group_data = viz1_df[viz1_df['Urban_Group'] == urban_group]
    
    fig1.add_trace(go.Bar(
        name=urban_group,
        x=group_data['Indicator'],
        y=group_data['Normalized_Score'],
        marker_color=colors_urban[i],
        marker_line_color='black',
        marker_line_width=1,
        hovertemplate=(
            '<b>%{x}</b><br>' +
            f'{urban_group}<br>' +
            'Normalized Score: %{y:.3f}<br>' +
            '<extra></extra>'
        ),
        customdata=group_data['Correlation']
    ))

# Add annotations for interpretation (arrows)
# Define which indicators are "lower is better" (inverted GPI scale)
higher_is_worse = ['internal peace', 'ongoing conflict', 'Neighbouring countries relations', 
                   'intensity of internal conflict', 'Political instability', 'overall score']

annotations = []
for i, indicator in enumerate(selected_indicators):
    if indicator in higher_is_worse:
        # Red down arrow for "lower is better"
        annotations.append(
            dict(
                x=i,
                y=1.05,
                text='↓ lower=better',
                showarrow=False,
                font=dict(size=10, color='#d62728'),
                xref='x',
                yref='paper'
            )
        )
    else:
        # Green up arrow for "higher is better"
        annotations.append(
            dict(
                x=i,
                y=1.05,
                text='↑ higher=better',
                showarrow=False,
                font=dict(size=10, color='#2ca02c'),
                xref='x',
                yref='paper'
            )
        )

fig1.update_layout(
    title={
        'text': 'Comparison of Normalized Indicator Scores by Urbanization Level<br><sub>Click on an indicator to see detailed analysis below</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'family': 'Arial, sans-serif'}
    },
    xaxis_title='Indicator',
    yaxis_title='Normalized Average Score (0-1)',
    barmode='group',
    hovermode='closest',
    legend=dict(
        title='Urbanization Level',
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    ),
    height=600,
    template='plotly_white',
    annotations=annotations,
    xaxis=dict(tickangle=-45),
    margin=dict(b=150, t=120)
)

print("✓ Visualization 1 created")

# ============================================================================
# VISUALIZATION 2: INTERACTIVE DEEP-DIVE SCATTER PLOTS
# ============================================================================
print("\nCreating Visualization 2: Deep-Dive Analysis...")

def create_detailed_viz(selected_indicator='overall score'):
    """
    Create a 3-panel scatter plot comparing:
    - All countries combined
    - Stable Urbanizers
    - Volatile Urbanizers
    """
    
    # Prepare data
    plot_data = df.dropna(subset=[urban_metric, selected_indicator, 'Country_Cluster']).copy()
    
    # Calculate correlations
    corr_all = plot_data[urban_metric].corr(plot_data[selected_indicator])
    
    stable_data = plot_data[plot_data['Country_Cluster'] == 'Stable Urbanizers']
    corr_stable = stable_data[urban_metric].corr(stable_data[selected_indicator]) if len(stable_data) > 0 else 0
    
    volatile_data = plot_data[plot_data['Country_Cluster'] == 'Volatile Urbanizers']
    corr_volatile = volatile_data[urban_metric].corr(volatile_data[selected_indicator]) if len(volatile_data) > 0 else 0
    
    # Determine y-axis label based on indicator type
    y_label = selected_indicator
    if selected_indicator == 'overall score':
        y_label = 'Global Peace Index (higher = worse)'
    elif selected_indicator in higher_is_worse:
        y_label = f'{selected_indicator} (higher = worse)'
    
    # Create subplot figure
    fig2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'<b>All Countries</b><br>r = {corr_all:.3f}',
            f'<b>Stable Urbanizers</b><br>r = {corr_stable:.3f}',
            f'<b>Volatile Urbanizers</b><br>r = {corr_volatile:.3f}'
        ),
        horizontal_spacing=0.08
    )
    
    # Panel 1: All Countries
    fig2.add_trace(
        go.Scatter(
            x=plot_data[urban_metric],
            y=plot_data[selected_indicator],
            mode='markers',
            marker=dict(
                color=COLORS['all_countries'],
                size=6,
                opacity=0.5,
                line=dict(width=0.5, color='black')
            ),
            name='All Countries',
            hovertemplate='<b>%{text}</b><br>Urban: %{x:.1f}%<br>' + y_label + ': %{y:.2f}<extra></extra>',
            text=plot_data['Country'],
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add regression line for all countries
    z = np.polyfit(plot_data[urban_metric], plot_data[selected_indicator], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_data[urban_metric].min(), plot_data[urban_metric].max(), 100)
    fig2.add_trace(
        go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            line=dict(color=COLORS['all_countries'], width=2),
            name='Trend',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Panel 2: Stable Urbanizers
    fig2.add_trace(
        go.Scatter(
            x=stable_data[urban_metric],
            y=stable_data[selected_indicator],
            mode='markers',
            marker=dict(
                color=COLORS['stable_cluster'],
                size=6,
                opacity=0.6,
                line=dict(width=0.5, color='black')
            ),
            name='Stable Urbanizers',
            hovertemplate='<b>%{text}</b><br>Urban: %{x:.1f}%<br>' + y_label + ': %{y:.2f}<extra></extra>',
            text=stable_data['Country'],
            showlegend=True
        ),
        row=1, col=2
    )
    
    # Add regression line for stable
    if len(stable_data) > 1:
        z_stable = np.polyfit(stable_data[urban_metric], stable_data[selected_indicator], 1)
        p_stable = np.poly1d(z_stable)
        x_stable = np.linspace(stable_data[urban_metric].min(), stable_data[urban_metric].max(), 100)
        fig2.add_trace(
            go.Scatter(
                x=x_stable,
                y=p_stable(x_stable),
                mode='lines',
                line=dict(color=COLORS['stable_cluster'], width=2),
                name='Trend',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Panel 3: Volatile Urbanizers
    fig2.add_trace(
        go.Scatter(
            x=volatile_data[urban_metric],
            y=volatile_data[selected_indicator],
            mode='markers',
            marker=dict(
                color=COLORS['volatile_cluster'],
                size=6,
                opacity=0.6,
                line=dict(width=0.5, color='black')
            ),
            name='Volatile Urbanizers',
            hovertemplate='<b>%{text}</b><br>Urban: %{x:.1f}%<br>' + y_label + ': %{y:.2f}<extra></extra>',
            text=volatile_data['Country'],
            showlegend=True
        ),
        row=1, col=3
    )
    
    # Add regression line for volatile
    if len(volatile_data) > 1:
        z_volatile = np.polyfit(volatile_data[urban_metric], volatile_data[selected_indicator], 1)
        p_volatile = np.poly1d(z_volatile)
        x_volatile = np.linspace(volatile_data[urban_metric].min(), volatile_data[urban_metric].max(), 100)
        fig2.add_trace(
            go.Scatter(
                x=x_volatile,
                y=p_volatile(x_volatile),
                mode='lines',
                line=dict(color=COLORS['volatile_cluster'], width=2),
                name='Trend',
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Update layout
    fig2.update_xaxes(title_text='Urban Population (%)', row=1, col=1)
    fig2.update_xaxes(title_text='Urban Population (%)', row=1, col=2)
    fig2.update_xaxes(title_text='Urban Population (%)', row=1, col=3)
    
    fig2.update_yaxes(title_text=y_label, row=1, col=1)
    fig2.update_yaxes(title_text='', row=1, col=2)
    fig2.update_yaxes(title_text='', row=1, col=3)
    
    # Determine if trend is improving or worsening
    if selected_indicator in higher_is_worse:
        trend_text_volatile = '↗ Worsens' if corr_volatile > 0 else '↘ Improves'
    else:
        trend_text_volatile = '↗ Improves' if corr_volatile > 0 else '↘ Worsens'
    
    fig2.update_layout(
        title={
            'text': f'<b>Deep-Dive Analysis: {selected_indicator}</b><br>' +
                   f'<sub>Simpson\'s Paradox: Global trend (r={corr_all:.3f}) masks divergent patterns</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5
        ),
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig2

# Create initial visualization with 'overall score'
fig2_initial = create_detailed_viz('overall score')

print("✓ Visualization 2 created")

# ============================================================================
# SAVE VISUALIZATIONS
# ============================================================================
print("\nSaving visualizations...")

# Save as HTML files
fig1.write_html('visualization_1_multi_indicator_comparison.html')
print("✓ Saved: visualization_1_multi_indicator_comparison.html")

fig2_initial.write_html('visualization_2_detailed_analysis.html')
print("✓ Saved: visualization_2_detailed_analysis.html")

# ============================================================================
# CREATE INTERACTIVE DASHBOARD WITH CALLBACKS
# ============================================================================
print("\nCreating interactive dashboard with Dash...")

try:
    from dash import Dash, dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    
    # Initialize Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # App layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Interactive Urbanization & Life Quality Analysis", 
                       className="text-center mb-4 mt-4"),
                html.P("Click on any indicator in the top chart to see detailed cluster analysis below",
                      className="text-center text-muted mb-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Visualization 1: Multi-Indicator Overview", className="mb-3"),
                dcc.Graph(id='indicator-comparison', figure=fig1, style={'height': '600px'})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Hr(className="my-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Visualization 2: Detailed Cluster Analysis", className="mb-3"),
                html.Div(id='selected-indicator-text', className="text-center mb-3"),
                dcc.Graph(id='detailed-analysis', figure=fig2_initial, style={'height': '500px'})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Hr(className="my-4"),
                html.Div([
                    html.H5("Key Insights:"),
                    html.Ul([
                        html.Li("Simpson's Paradox: Global correlations can mask opposite patterns in subgroups"),
                        html.Li("Stable vs Volatile Urbanizers show fundamentally different relationships"),
                        html.Li("Policy interventions must account for country development context"),
                        html.Li("Urbanization impact depends on HOW you urbanize, not just THAT you urbanize")
                    ])
                ], className="text-muted p-3 bg-light rounded")
            ], width=12)
        ])
    ], fluid=True)
    
    # Callback to update detailed visualization based on clicked indicator
    @app.callback(
        [Output('detailed-analysis', 'figure'),
         Output('selected-indicator-text', 'children')],
        [Input('indicator-comparison', 'clickData')]
    )
    def update_detailed_viz(clickData):
        # Default to 'overall score' if no click
        selected = 'overall score'
        
        if clickData is not None:
            try:
                selected = clickData['points'][0]['x']
            except:
                pass
        
        # Create updated figure
        new_fig = create_detailed_viz(selected)
        
        # Create text display
        text_display = html.Div([
            html.Strong(f"Currently viewing: ", style={'fontSize': '16px'}),
            html.Span(selected, style={'fontSize': '16px', 'color': '#E41A1C', 'fontWeight': 'bold'})
        ])
        
        return new_fig, text_display
    
    print("✓ Interactive dashboard created")
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nTo run the interactive dashboard:")
    print("  python interactive_urbanization_analysis.py")
    print("\nThen open your browser to: http://127.0.0.1:8050/")
    print("\nStandalone HTML files also saved:")
    print("  • visualization_1_multi_indicator_comparison.html")
    print("  • visualization_2_detailed_analysis.html")
    print("="*70)
    
    # Run the app
    if __name__ == '__main__':
        app.run_server(debug=True, port=8050)
        
except ImportError:
    print("\n⚠ Dash not installed. Standalone HTML files created.")
    print("\nTo enable interactive dashboard, install Dash:")
    print("  pip install dash dash-bootstrap-components")
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nStandalone HTML files saved:")
    print("  • visualization_1_multi_indicator_comparison.html")
    print("  • visualization_2_detailed_analysis.html")
    print("\nThese can be opened directly in a web browser.")
    print("="*70)
