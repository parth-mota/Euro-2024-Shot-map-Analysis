import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import seaborn as sns
import numpy as np
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap

# Streamlit page config
st.set_page_config(layout="wide", page_title="EURO 2024 Shot Map", page_icon="⚽")

# Theme with better contrast
st.markdown("""
    <style>
    body { background-color: #1e2130; color: #e0e0e0; }
    .stApp { background-color: #1e2130; }
    .metric-label, .metric-value { color: #e0e0e0 !important; }
    .css-1v3fvcr { background-color: #1e2130 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #2c3152;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b5288;
        color: white;
    }
    /* Fix chart container height */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        min-height: 500px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ EURO 2024 Shot Map Dashboard")
st.markdown("Filter by team, player, shot type, and explore insights interactively.")

# Load dataset - using st.cache_data for better performance
@st.cache_data
def load_data():
    try:
        # Replace with relative path or URL for GitHub
        df = pd.read_csv(r'C:\Users\LENOVO\Downloads\dataset_euros24.csv')
        df = df[df['type'] == 'Shot'].reset_index(drop=True)
        # Safely parse location data
        df['location'] = df['location'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        # Standardize shot outcomes to ensure consistent naming
        shot_outcome_mapping = {
            'Goal': 'Goal',
            'Saved': 'Saved',
            'Off T': 'Off Target',
            'Off Target': 'Off Target',
            'Blocked': 'Blocked',
            'Post': 'Post',
            'Wayward': 'Off Target',
            'Saved Off Target': 'Saved',
            'Saved to Post': 'Saved'
        }
        # Apply mapping and ensure any unmapped values get a default
        df['shot_outcome'] = df['shot_outcome'].map(lambda x: shot_outcome_mapping.get(x, 'Other'))
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Dataset could not be loaded. Please check the file path.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")
teams = sorted(df['team'].unique())
team = st.sidebar.selectbox("Select Team", teams)

player_list = sorted(df[df['team'] == team]['player'].unique())
player = st.sidebar.selectbox("Select Player", player_list)

# Check if shot_body_part exists and handle gracefully
if 'shot_body_part' in df.columns:
    body_parts = sorted(df['shot_body_part'].dropna().unique())
    selected_parts = st.sidebar.multiselect("Shot Type", body_parts, default=body_parts)
else:
    st.sidebar.info("Shot body part data not available")
    selected_parts = []

# Filter data
filtered = df[(df['team'] == team) & (df['player'] == player)]
if selected_parts and 'shot_body_part' in df.columns:
    filtered = filtered[filtered['shot_body_part'].isin(selected_parts)]

# Tabs
tabs = st.tabs(["Shot Map", "Stats Overview", "Player Comparison", "Heatmap"])

# Enhanced color palette with better contrast and added "Other" category
outcome_colors = {
    'Goal': '#e63946',          # Vibrant red for goals
    'Saved': '#45b6fe',         # Brighter blue for saved shots
    'Blocked': '#f9c74f',       # Brighter yellow for blocked shots
    'Off Target': '#645394',    # Richer purple for off-target
    'Post': '#ff9e00',          # Orange for hitting the post
    'Other': '#8a8a8a'          # Gray for any other outcomes
}

# Custom colormap for heatmap - deeper colors
heat_colors = ["#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]
custom_cmap = LinearSegmentedColormap.from_list("custom_heat", heat_colors)

# --- 1. SHOT MAP ---
with tabs[0]:
    st.subheader(f"🎯 Shot Map - {player}")
    
    pitch_fig_col1, pitch_fig_col2 = st.columns([3, 1])
    
    with pitch_fig_col1:
        # Create a more aesthetically pleasing pitch
        pitch = VerticalPitch(
            pitch_type='statsbomb', 
            line_zorder=2, 
            pitch_color='#2c5530',  # Deep green
            line_color='#ffffff',   # White lines
            half=True
        )
        
        fig, ax = pitch.draw(figsize=(10, 8))
        
        # Plot shots with proper sizing and colors - ensure all shots have colors
        min_size = 100  # Minimum size for visibility
        for _, row in filtered.iterrows():
            try:
                x, y = row['location']
                xg_value = row['shot_statsbomb_xg'] if not pd.isna(row['shot_statsbomb_xg']) else 0.05
                
                # Get color, defaulting to "Other" if no match
                shot_color = outcome_colors.get(row['shot_outcome'], outcome_colors['Other'])
                
                # Ensure minimum size and proper color
                marker_size = max(min_size, 800 * xg_value)  # Bigger markers for better visibility
                
                pitch.scatter(
                    x=float(x),
                    y=float(y),
                    ax=ax,
                    s=marker_size,  # Larger minimum size for visibility
                    color=shot_color,
                    edgecolors='white',
                    linewidth=1.5,  # Thicker border for definition
                    alpha=0.9,  # More opaque
                    zorder=3
                )
            except (TypeError, ValueError) as e:
                st.sidebar.write(f"Error with shot data: {e}")
                continue  # Skip problematic data points
        
        # Create legend with all possible outcomes that might appear
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=out, 
                     markerfacecolor=col, markersize=10)  # Bigger legend markers
            for out, col in outcome_colors.items()
            if out in filtered['shot_outcome'].unique() or out == 'Other'  # Only include relevant outcomes
        ]
        ax.legend(
            handles=legend_elements, 
            loc='lower right', 
            facecolor='#333333', 
            framealpha=0.8, 
            edgecolor='white', 
            labelcolor='white',
            fontsize=10
        )
        
        st.pyplot(fig)
    
    with pitch_fig_col2:
        st.markdown("#### Shot Legend")
        st.markdown("The size of each point represents the Expected Goals (xG) value.")
        st.markdown("#### Color Code:")
        for outcome, color in outcome_colors.items():
            # Only show color codes for outcomes in the data or the "Other" category
            if outcome in filtered['shot_outcome'].unique() or outcome == 'Other':
                st.markdown(f"<div style='background-color:{color}; width:20px; height:20px; display:inline-block; margin-right:10px;'></div> {outcome}", unsafe_allow_html=True)

# --- 2. STATS OVERVIEW ---
with tabs[1]:
    st.subheader("📊 Detailed Stats")
    
    total_shots = len(filtered)
    
    if total_shots > 0:
        goals = (filtered['shot_outcome'] == 'Goal').sum()
        xg_total = round(filtered['shot_statsbomb_xg'].sum(), 3)
        avg_xg = round(filtered['shot_statsbomb_xg'].mean(), 3)
        on_target = len(filtered[filtered['shot_outcome'].isin(['Goal', 'Saved'])])
        accuracy = round(on_target / total_shots * 100, 1) if total_shots else 0
        conversion = round(goals / total_shots * 100, 1) if total_shots else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Shots", total_shots)
        col2.metric("Goals", goals)
        col3.metric("Total xG", xg_total)
        col4.metric("Accuracy %", f"{accuracy}%")
        col5.metric("Conversion Rate", f"{conversion}%")
        
        st.markdown("---")
        st.markdown("#### Shot Outcome Distribution")
        
        outcome_counts = filtered['shot_outcome'].value_counts().reset_index()
        outcome_counts.columns = ['Outcome', 'Count']
        
        fig2 = px.bar(
            outcome_counts, 
            x='Outcome', 
            y='Count',
            color='Outcome',
            color_discrete_map=outcome_colors,
            text='Count'
        )
        
        # Improve chart aesthetics
        fig2.update_layout(
            plot_bgcolor='#252e3f',
            paper_bgcolor='#252e3f',
            font_color='#e0e0e0',
            margin=dict(t=50, b=50),
            height=400,  # Fixed height
            xaxis=dict(
                title=None,
                gridcolor='#3b4559'
            ),
            yaxis=dict(
                title="Number of Shots",
                gridcolor='#3b4559'
            ),
            legend_title_text=None
        )
        
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No shot data available for {player}")

# --- 3. PLAYER COMPARISON ---
with tabs[2]:
    st.subheader("👊 Side-by-Side Comparison")
    
    # Get top scorers for better default comparison
    top_scorers = df.groupby('player').apply(
        lambda x: (x['shot_outcome'] == 'Goal').sum()
    ).sort_values(ascending=False).head(10).index.tolist()
    
    # Add current player to list if not already there
    if player not in top_scorers:
        top_scorers.insert(0, player)
    
    compare_players = st.multiselect(
        "Compare Up To 5 Players (any country)", 
        df['player'].unique(), 
        default=[player],
        max_selections=5
    )
    
    if compare_players:
        compare_data = df[df['player'].isin(compare_players)]
        
        # Calculate statistics with error handling
        stats = compare_data.groupby('player').agg(
            Goals=('shot_outcome', lambda x: (x == 'Goal').sum()),
            Total_Shots=('shot_outcome', 'count'),
            Total_xG=('shot_statsbomb_xg', 'sum'),
            Avg_xG=('shot_statsbomb_xg', 'mean'),
            On_Target_Pct=('shot_outcome', lambda x: sum(x.isin(['Goal', 'Saved'])) / len(x) * 100 if len(x) > 0 else 0)
        ).round(2).reset_index()
        
        # Convert to long format for plotting
        stats_long = stats.melt(id_vars='player', var_name='Metric', value_name='Value')
        
        # Define vibrant colors for players
        player_colors = px.colors.qualitative.Bold
        
        # Create comparison chart with improved layout
        fig3 = px.bar(
            stats_long, 
            x='Metric', 
            y='Value', 
            color='player', 
            barmode='group', 
            text='Value',
            color_discrete_sequence=player_colors
        )
        
        # Fix chart height and layout
        fig3.update_layout(
            plot_bgcolor='#252e3f',
            paper_bgcolor='#252e3f',
            font_color='#e0e0e0',
            margin=dict(l=50, r=50, t=50, b=80),  # Increased bottom margin
            height=500,  # Fixed height to prevent cutoff
            xaxis=dict(
                title=None,
                gridcolor='#3b4559',
                tickangle=0  # Horizontal labels
            ),
            yaxis=dict(
                title="Value",
                gridcolor='#3b4559'
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=-0.25,  # Position below chart
                xanchor="center",
                x=0.5
            )
        )
        
        # Better text positioning
        fig3.update_traces(
            textposition='outside',
            textfont=dict(size=10)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Select at least one player to compare")

# --- 4. HEATMAP ---
with tabs[3]:
    st.subheader("🔥 Shot Location Heatmap")
    
    if len(filtered) >= 3:  # Need minimum points for a meaningful heatmap
        # Extract location coordinates safely
        x_coords = []
        y_coords = []
        
        for loc in filtered['location']:
            try:
                if isinstance(loc, list) and len(loc) >= 2:
                    x_coords.append(float(loc[0]))
                    y_coords.append(float(loc[1]))
            except (TypeError, ValueError, IndexError):
                continue
        
        if len(x_coords) >= 3:  # Still need minimum points after filtering
            heat_pitch = VerticalPitch(
                pitch_type='statsbomb', 
                pitch_color='#2c5530', 
                line_color='white', 
                half=True
            )
            
            fig4, ax4 = heat_pitch.draw(figsize=(10, 8))
            
            # Use KDE plot with proper levels and colors
            try:
                heat_pitch.kdeplot(
                    x=x_coords,
                    y=y_coords,
                    ax=ax4,
                    cmap=custom_cmap,
                    fill=True,
                    levels=12,  # Reduced for clarity
                    alpha=0.7,
                    thresh=0.05
                )
                
                # Plot shots as larger points for reference
                heat_pitch.scatter(
                    x=x_coords,
                    y=y_coords,
                    ax=ax4,
                    s=30,  # Larger points
                    color='white',
                    alpha=0.5,
                    edgecolors='black',
                    linewidth=0.5,
                    zorder=2
                )
                
                st.pyplot(fig4)
                
                st.markdown("#### Heatmap Interpretation")
                st.markdown("""
                This heatmap shows the density of shot locations. 
                - Darker red areas indicate higher concentrations of shots
                - Yellow areas show fewer shots
                - White dots represent individual shot locations
                """)
                
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")
                st.info("Try selecting a player with more shot data.")
        else:
            st.info(f"Not enough valid shot locations for {player} to create a heatmap.")
    else:
        st.info(f"Not enough shots for {player} to create a meaningful heatmap.")