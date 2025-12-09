"""
Flask Web Application for NBA Momentum HMM Visualization

Provides interactive UI for:
- Game selection
- Momentum timeline visualization
- Win probability curves
- Transition matrix and state statistics
"""
import sys
sys.path.append('..')

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and data
print("Loading model and data...")
with open('../final_momentum_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Get HMM object
hmm = model_data['hmm']

df = pd.read_csv('../possessions_multi_season_with_players.csv')
print(f"Loaded {len(df):,} possessions from {df['gameId'].nunique()} games")

# Add state predictions if not already present
if 'state' not in df.columns:
    print("Computing momentum states...")
    from train_hmm_enhanced import create_enhanced_features
    df = create_enhanced_features(df)

    features = model_data['features']
    X, lengths = hmm.prepare_sequences(df, features)
    states = hmm.predict_states(X, lengths)
    df['state'] = states

# Train win probability model
print("Training win probability model...")
win_prob_model = None
win_prob_scaler = None

try:
    # Calculate final scores for each game
    game_outcomes = {}
    for game_id in df['gameId'].unique():
        game_df = df[df['gameId'] == game_id].copy()
        game_df = game_df.sort_values('possession_idx')

        # Calculate final scores
        team_score = game_df['pointsScored'].sum()
        opp_score = game_df['pointsAllowed'].sum()
        game_outcomes[game_id] = 1 if team_score > opp_score else 0

    # Add team_won column
    df['team_won'] = df['gameId'].map(game_outcomes)

    # Prepare features for win probability prediction
    # Features: state (one-hot), score differential, time remaining, period
    df_win = df.copy()
    df_win['teamScore'] = df_win.groupby('gameId')['pointsScored'].cumsum()
    df_win['oppScore'] = df_win.groupby('gameId')['pointsAllowed'].cumsum()
    df_win['scoreDiff'] = df_win['teamScore'] - df_win['oppScore']
    df_win['timeRemaining'] = (4 - df_win['period']) * 12 + df_win['clockStart'] / 60

    # One-hot encode states
    for state_num in range(model_data['n_states']):
        df_win[f'state_{state_num}'] = (df_win['state'] == state_num).astype(int)

    # Build feature matrix
    feature_cols = [f'state_{i}' for i in range(model_data['n_states'])] + ['scoreDiff', 'timeRemaining', 'period']
    X_win = df_win[feature_cols].values
    y_win = df_win['team_won'].values

    # Remove any NaN values
    valid_idx = ~np.isnan(X_win).any(axis=1) & ~np.isnan(y_win)
    X_win = X_win[valid_idx]
    y_win = y_win[valid_idx]

    if len(X_win) > 100:  # Only train if we have enough data
        # Scale features
        win_prob_scaler = StandardScaler()
        X_win_scaled = win_prob_scaler.fit_transform(X_win)

        # Train logistic regression
        win_prob_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        win_prob_model.fit(X_win_scaled, y_win)

        # Calculate training accuracy
        train_acc = win_prob_model.score(X_win_scaled, y_win)
        print(f"✓ Win probability model trained on {len(X_win):,} possessions")
        print(f"  Training accuracy: {train_acc:.4f}")
    else:
        print(f"⚠️  Not enough data ({len(X_win)} samples) to train win probability model")

except Exception as e:
    print(f"⚠️  Could not train win probability model: {e}")
    import traceback
    traceback.print_exc()

# Load team names from JSON file
game_teams_file = '../game_teams.json'
game_teams = {}

if os.path.exists(game_teams_file):
    print(f"Loading team names from {game_teams_file}...")
    with open(game_teams_file, 'r') as f:
        game_teams = json.load(f)
    print(f"✓ Loaded team names for {len(game_teams)} games")
else:
    print(f"⚠️  Team names file not found. Run: python fetch_game_teams.py")

# Get unique games with metadata
games_list = []
for game_id in df['gameId'].unique():
    game_df = df[df['gameId'] == game_id]

    # Get team names from loaded data
    teams = game_teams.get(str(game_id))

    if teams and teams['home'] != 'N/A':
        game_desc = f"{teams['away']} @ {teams['home']}"
    else:
        game_desc = f"Game {game_id}"

    games_list.append({
        'gameId': str(game_id),
        'season': str(game_df.iloc[0]['season']) if 'season' in game_df.columns else 'N/A',
        'n_possessions': len(game_df),
        'description': game_desc,
        'homeTeam': teams['home'] if teams else 'N/A',
        'awayTeam': teams['away'] if teams else 'N/A'
    })

games_list = sorted(games_list, key=lambda x: x['gameId'])
print(f"✓ Loaded {len(games_list)} games")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/games')
def get_games():
    """Get list of available games"""
    return jsonify(games_list)

@app.route('/api/model-stats')
def get_model_stats():
    """Get model statistics"""
    # Get transition matrix from HMM object
    transition_matrix = hmm.model.transmat_

    stats = {
        'n_states': model_data['n_states'],
        'auc_baseline': model_data.get('auc_baseline', 0.5),
        'auc_final': model_data.get('auc_final', model_data.get('auc_momentum', 0)),
        'improvement': model_data.get('improvement', 0),
        'anova_pvalue': model_data.get('anova_pvalue', 0),
        'transition_matrix': transition_matrix.tolist()
    }
    return jsonify(stats)

@app.route('/api/state-stats')
def get_state_stats():
    """Get statistics for each momentum state"""
    state_stats = []

    for state_num in range(model_data['n_states']):
        state_data = df[df['state'] == state_num]

        stats = {
            'state': int(state_num),
            'count': int(len(state_data)),
            'pct': float(len(state_data) / len(df) * 100),
            'scoring_rate': float((state_data['pointsScored'] > 0).mean() * 100),
            'avg_points': float(state_data['pointsScored'].mean()),
            'name': ['COLD', 'NEUTRAL', 'HOT'][state_num] if state_num < 3 else f'State {state_num}'
        }
        state_stats.append(stats)

    # Sort by scoring rate to assign names correctly
    state_stats = sorted(state_stats, key=lambda x: x['scoring_rate'])
    for i, stats in enumerate(state_stats):
        stats['name'] = ['COLD', 'NEUTRAL', 'HOT'][i]

    return jsonify(state_stats)

@app.route('/api/game/<game_id>')
def get_game_data(game_id):
    """Get detailed data for a specific game"""
    game_df = df[df['gameId'] == int(game_id)].copy()

    if len(game_df) == 0:
        return jsonify({'error': 'Game not found'}), 404

    # Sort by possession index
    game_df = game_df.sort_values('possession_idx')

    # Calculate cumulative scores
    game_df['teamScore'] = game_df['pointsScored'].cumsum()
    game_df['oppScore'] = game_df['pointsAllowed'].cumsum()
    game_df['scoreDiff'] = game_df['teamScore'] - game_df['oppScore']
    game_df['timeRemaining'] = (4 - game_df['period']) * 12 + game_df['clockStart'] / 60

    # Use ML model if available, otherwise fall back to simple formula
    if win_prob_model is not None and win_prob_scaler is not None:
        # One-hot encode states
        for state_num in range(model_data['n_states']):
            game_df[f'state_{state_num}'] = (game_df['state'] == state_num).astype(int)

        # Build feature matrix
        feature_cols = [f'state_{i}' for i in range(model_data['n_states'])] + ['scoreDiff', 'timeRemaining', 'period']
        X_pred = game_df[feature_cols].values

        # Scale and predict
        X_pred_scaled = win_prob_scaler.transform(X_pred)
        win_probs_ml = win_prob_model.predict_proba(X_pred_scaled)[:, 1]

        # Blend ML model with deterministic formula in last 5 possessions
        total_possessions = len(game_df)
        win_probs = []
        for idx in range(total_possessions):
            possessions_remaining = total_possessions - idx - 1
            score_diff = game_df.iloc[idx]['scoreDiff']

            if possessions_remaining == 0:
                # Last possession: deterministic based on score
                final_prob = 1.0 if score_diff > 0 else (0.5 if score_diff == 0 else 0.0)
            elif possessions_remaining < 5:
                # Blend with simple formula
                simple_prob = 1 / (1 + np.exp(-score_diff / (1 + game_df.iloc[idx]['timeRemaining'] / 10)))
                blend_weight = (5 - possessions_remaining) / 5.0
                final_prob = (1 - blend_weight) * win_probs_ml[idx] + blend_weight * simple_prob
            else:
                # Use pure ML prediction
                final_prob = win_probs_ml[idx]

            win_probs.append(final_prob)

        game_df['winProb'] = win_probs
    else:
        # Fallback to simple formula
        game_df['winProb'] = 1 / (1 + np.exp(-game_df['scoreDiff'] / (1 + game_df['timeRemaining'] / 10)))

    team_score = int(game_df['teamScore'].iloc[-1])
    opp_score = int(game_df['oppScore'].iloc[-1])

    # Prepare data for frontend
    possessions = []
    for idx, row in game_df.iterrows():
        possessions.append({
            'possession': int(row['possession_idx']),
            'period': int(row['period']),
            'clock': float(row['clockStart']),
            'state': int(row['state']),
            'pointsScored': int(row['pointsScored']),
            'pointsAllowed': int(row['pointsAllowed']),
            'teamScore': int(row['teamScore']),
            'oppScore': int(row['oppScore']),
            'scoreDiff': int(row['scoreDiff']),
            'winProb': float(row['winProb']),
            'description': str(row.get('description', '')),
            'result': str(row.get('result', ''))
        })

    game_info = {
        'gameId': str(game_id),
        'season': str(game_df.iloc[0]['season']) if 'season' in game_df.columns else 'N/A',
        'finalScore': f"{int(team_score)}-{int(opp_score)}",
        'possessions': possessions
    }

    return jsonify(game_info)

@app.route('/api/momentum-timeline/<game_id>')
def get_momentum_timeline(game_id):
    """Get Plotly figure for momentum timeline"""
    game_df = df[df['gameId'] == int(game_id)].copy()

    if len(game_df) == 0:
        return jsonify({'error': 'Game not found'}), 404

    game_df = game_df.sort_values('possession_idx')

    # Define colors for states
    state_colors = {0: '#3498db', 1: '#95a5a6', 2: '#e74c3c'}  # Blue, Gray, Red
    state_names = {0: 'COLD', 1: 'NEUTRAL', 2: 'HOT'}

    # Map states by scoring rate
    state_scoring = []
    for state_num in game_df['state'].unique():
        state_data = game_df[game_df['state'] == state_num]
        scoring_rate = (state_data['pointsScored'] > 0).mean()
        state_scoring.append((state_num, scoring_rate))

    state_scoring = sorted(state_scoring, key=lambda x: x[1])
    state_mapping = {state_num: i for i, (state_num, _) in enumerate(state_scoring)}

    game_df['state_mapped'] = game_df['state'].map(state_mapping)

    # Build traces manually to avoid binary encoding
    traces = []

    # Add state background
    for state_num in [0, 1, 2]:
        state_data = game_df[game_df['state_mapped'] == state_num]
        if len(state_data) > 0:
            traces.append({
                'x': state_data['possession_idx'].tolist(),
                'y': [state_num] * len(state_data),
                'type': 'scatter',
                'mode': 'markers',
                'marker': {
                    'size': 15,
                    'color': state_colors[state_num],
                    'symbol': 'square',
                },
                'name': state_names[state_num],
                'hovertemplate': f'<b>{state_names[state_num]}</b><br>Possession: %{{x}}<extra></extra>'
            })

    # Add scoring events
    scoring_data = game_df[game_df['pointsScored'] > 0]
    if len(scoring_data) > 0:
        traces.append({
            'x': scoring_data['possession_idx'].tolist(),
            'y': scoring_data['state_mapped'].tolist(),
            'type': 'scatter',
            'mode': 'markers',
            'marker': {
                'size': 12,
                'color': 'gold',
                'symbol': 'star',
                'line': {'color': 'black', 'width': 1}
            },
            'name': 'Scored',
            'hovertemplate': '<b>SCORED %{text}</b><br>Possession: %{x}<extra></extra>',
            'text': [f"{int(pts)} pts" for pts in scoring_data['pointsScored']]
        })

    fig_data = {
        'data': traces,
        'layout': {
            'title': f'Momentum Timeline - Game {game_id}',
            'xaxis': {'title': 'Possession Number'},
            'yaxis': {
                'tickmode': 'array',
                'tickvals': [0, 1, 2],
                'ticktext': ['COLD', 'NEUTRAL', 'HOT'],
                'title': 'Momentum State'
            },
            'hovermode': 'closest',
            'height': 400,
            'template': 'plotly_white'
        }
    }

    return jsonify(fig_data)

@app.route('/api/win-probability/<game_id>')
def get_win_probability(game_id):
    """Get Plotly figure for win probability curve"""
    game_df = df[df['gameId'] == int(game_id)].copy()

    if len(game_df) == 0:
        return jsonify({'error': 'Game not found'}), 404

    game_df = game_df.sort_values('possession_idx')

    # Calculate cumulative scores
    game_df['teamScore'] = game_df['pointsScored'].cumsum()
    game_df['oppScore'] = game_df['pointsAllowed'].cumsum()
    game_df['scoreDiff'] = game_df['teamScore'] - game_df['oppScore']
    game_df['timeRemaining'] = (4 - game_df['period']) * 12 + game_df['clockStart'] / 60

    # Use ML model if available, otherwise fall back to simple formula
    if win_prob_model is not None and win_prob_scaler is not None:
        # One-hot encode states
        for state_num in range(model_data['n_states']):
            game_df[f'state_{state_num}'] = (game_df['state'] == state_num).astype(int)

        # Build feature matrix
        feature_cols = [f'state_{i}' for i in range(model_data['n_states'])] + ['scoreDiff', 'timeRemaining', 'period']
        X_pred = game_df[feature_cols].values

        # Scale and predict
        X_pred_scaled = win_prob_scaler.transform(X_pred)
        win_probs_ml = win_prob_model.predict_proba(X_pred_scaled)[:, 1]

        # Blend ML model with deterministic formula in last 5 possessions
        total_possessions = len(game_df)
        win_probs = []
        for idx in range(total_possessions):
            possessions_remaining = total_possessions - idx - 1
            score_diff = game_df.iloc[idx]['scoreDiff']

            if possessions_remaining == 0:
                # Last possession: deterministic based on score
                final_prob = 1.0 if score_diff > 0 else (0.5 if score_diff == 0 else 0.0)
            elif possessions_remaining < 5:
                # Blend with simple formula
                simple_prob = 1 / (1 + np.exp(-score_diff / (1 + game_df.iloc[idx]['timeRemaining'] / 10)))
                blend_weight = (5 - possessions_remaining) / 5.0
                final_prob = (1 - blend_weight) * win_probs_ml[idx] + blend_weight * simple_prob
            else:
                # Use pure ML prediction
                final_prob = win_probs_ml[idx]

            win_probs.append(final_prob)

        game_df['winProb'] = win_probs
    else:
        # Fallback to simple formula
        game_df['winProb'] = 1 / (1 + np.exp(-game_df['scoreDiff'] / (1 + game_df['timeRemaining'] / 10)))

    # Build data for chart
    scores_df = game_df[['possession_idx', 'winProb']].copy()
    scores_df.columns = ['possession', 'winProb']

    # Build JSON directly to avoid binary encoding
    fig_data = {
        'data': [{
            'x': scores_df['possession'].tolist(),
            'y': (scores_df['winProb'] * 100).tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'fill': 'tozeroy',
            'line': {'color': '#2ecc71', 'width': 3},
            'name': 'Win Probability',
            'hovertemplate': '<b>Possession %{x}</b><br>Win Prob: %{y:.1f}%<extra></extra>'
        }],
        'layout': {
            'title': f'Win Probability - Game {game_id}',
            'xaxis': {'title': 'Possession Number'},
            'yaxis': {'title': 'Win Probability (%)', 'range': [0, 100]},
            'hovermode': 'x unified',
            'height': 400,
            'template': 'plotly_white',
            'shapes': [{
                'type': 'line',
                'x0': 0,
                'x1': 1,
                'xref': 'x domain',
                'y0': 50,
                'y1': 50,
                'yref': 'y',
                'line': {'color': 'gray', 'dash': 'dash'},
                'opacity': 0.5
            }]
        }
    }

    return jsonify(fig_data)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("NBA Momentum HMM Visualization Server")
    print("="*80)
    print(f"\nModel loaded: {model_data['n_states']} states")
    print(f"AUC: {model_data['auc_final']:.4f}")
    print(f"Games available: {len(games_list)}")
    print("\nStarting server at http://localhost:5000")
    print("="*80 + "\n")

    app.run(debug=True, port=5000)
