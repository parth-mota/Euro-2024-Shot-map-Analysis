import json
import pandas as pd
import streamlit as st
from mplsoccer import VerticalPitch

st.title("Euros 2024 Shot Map")
st.subheader("Filter to any team/player to see all their shots taken!")

# Load dataset
data_path = 'C:/Users/Lenovo/Downloads/euros_2024_shot_map.csv'
df = pd.read_csv(data_path)
df = df[df['type'] == 'Shot'].reset_index(drop=True)
df['location'] = df['location'].apply(json.loads)

def filter_data(df: pd.DataFrame, team: str, player: str):
    if team:
        df = df[df['team'] == team]
    if player:
        df = df[df['player'] == player]
    return df

def plot_shots(df, ax, pitch):
    for x in df.to_dict(orient='records'):
        pitch.scatter(
            x=float(x['location'][0]),
            y=float(x['location'][1]),
            ax=ax,
            s=1000 * x['shot_statsbomb_xg'],
            color='green' if x['shot_outcome'] == 'Goal' else 'white',
            edgecolors='black',
            alpha=1 if x['type'] == 'goal' else .5,
            zorder=2 if x['type'] == 'goal' else 1
        )

def plot_shot_outcomes(df, ax, pitch):
    outcomes = df['shot_outcome'].value_counts()
    colors = plt.cm.Set1(np.linspace(0, 1, len(outcomes)))
    for outcome, count, color in zip(outcomes.index, outcomes.values, colors):
        subset = df[df['shot_outcome'] == outcome]
        pitch.scatter(
            x=subset['location'].apply(lambda loc: float(loc[0])),
            y=subset['location'].apply(lambda loc: float(loc[1])),
            ax=ax,
            color=color,
            edgecolors='black',
            alpha=0.7,
            s=100,
            label=f'{outcome} ({count})'
        )
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))

def plot_heatmap(df, ax, pitch):
    from scipy.stats import gaussian_kde
    xy = np.vstack([df['location'].apply(lambda loc: float(loc[0])), df['location'].apply(lambda loc: float(loc[1]))])
    z = gaussian_kde(xy)(xy)
    scatter = ax.scatter(df['location'].apply(lambda loc: float(loc[0])), df['location'].apply(lambda loc: float(loc[1])), c=z, s=50, cmap='hot', edgecolor='black')
    plt.colorbar(scatter, label='Density')

def plot_xg(df, ax, pitch):
    scatter = pitch.scatter(
        x=df['location'].apply(lambda loc: float(loc[0])),
        y=df['location'].apply(lambda loc: float(loc[1])),
        ax=ax,
        s=100,
        c=df['shot_statsbomb_xg'],
        cmap='YlOrRd',
        edgecolor='black',
        vmin=0,
        vmax=1
    )
    plt.colorbar(scatter, label='Expected Goals (xG)')

# Filter the dataframe
team = st.selectbox("Select a team", df['team'].sort_values().unique(), index=None)
player = st.selectbox("Select a player", df[df['team'] == team]['player'].sort_values().unique(), index=None)
filtered_df = filter_data(df, team, player)

# Create a selectbox for plot type selection
plot_types = ['Shot Map', 'Shot Outcomes', 'Heatmap', 'Shot Statsbomb xG']
selected_plot = st.selectbox('Select a Plot Type', plot_types)

# Create the pitch
pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f0f0f0', line_color='black', half=True)
fig, ax = pitch.draw(figsize=(10, 10))

# Plot based on selected plot type
if selected_plot == 'Shot Map':
    plot_shots(filtered_df, ax, pitch)
    ax.set_title(f'Shot Map for {player}')
elif selected_plot == 'Shot Outcomes':
    plot_shot_outcomes(filtered_df, ax, pitch)
    ax.set_title(f'Shot Outcomes for {player}')
elif selected_plot == 'Heatmap':
    pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#f0f0f0', line_color='black', half=False)
    fig, ax = pitch.draw(figsize=(10, 10))
    plot_heatmap(filtered_df, ax, pitch)
    ax.set_title(f'Shot Heatmap for {player}')
elif selected_plot == 'Shot Statsbomb xG':
    plot_xg(filtered_df, ax, pitch)
    ax.set_title(f'Shot xG for {player}')

st.pyplot(fig)