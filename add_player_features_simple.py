"""
Simple Player Features - Using Existing Data
Uses rolling statistics from the play-by-play data itself
No external API calls needed!
"""
import pandas as pd
import numpy as np

def add_player_rolling_stats(possessions_file, output_file):
    """
    Add player features using rolling statistics from the game data itself

    Features added:
    - Player's rolling FG% (last 10 possessions)
    - Player's recent scoring rate (last 5 possessions)
    - Player's hot hand indicator (made last shot)
    - Team's shooting streak
    """
    print("="*60)
    print("ADDING PLAYER FEATURES (FROM EXISTING DATA)")
    print("="*60)

    # Load possession data
    print(f"\nLoading possessions from {possessions_file}...")
    poss_df = pd.read_csv(possessions_file)
    print(f"Loaded {len(poss_df):,} possessions")

    # Load play-by-play data
    print("\nLoading play-by-play data...")
    pbp_df = pd.read_csv('v3_multi_season.csv')
    print(f"Loaded {len(pbp_df):,} play-by-play events")

    # Feature 1: Identify primary player for each possession
    # Match possessions to play-by-play events
    print("\nMatching possessions to players...")

    # Merge on gameId and approximate time matching
    poss_df['possession_idx'] = poss_df.index

    # For simplicity, use the first event in each possession window
    # Group pbp by game and period
    pbp_df_sorted = pbp_df.sort_values(['gameId', 'period', 'clock'])

    # Create shooting events only (field goals + free throws)
    shooting_events = pbp_df[
        (pbp_df['isFieldGoal'] == 1) &
        (pbp_df['personId'].notna())
    ].copy()

    print(f"Found {len(shooting_events):,} shooting events with player IDs")

    # Feature 2: Calculate rolling player statistics
    print("\nCalculating player rolling statistics...")

    # For each player, calculate their rolling FG%
    shooting_events = shooting_events.sort_values(['personId', 'gameId', 'period', 'clock'])
    shooting_events['made_shot'] = (shooting_events['shotResult'] == 'Made').astype(int)

    # Rolling window features (last 10 shots per player)
    shooting_events['player_rolling_fg_pct'] = (
        shooting_events.groupby('personId')['made_shot']
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Hot hand: made last shot
    shooting_events['player_hot_hand'] = (
        shooting_events.groupby('personId')['made_shot']
        .shift(1)
        .fillna(0)
    )

    # Recent scoring (last 5 shots)
    shooting_events['player_recent_makes'] = (
        shooting_events.groupby('personId')['made_shot']
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    print("✓ Calculated rolling player stats")

    # Feature 3: Team-level features (from existing possession data)
    print("\nCalculating team-level features...")

    # Team shooting streak (consecutive scoring possessions)
    poss_df['scored_possession'] = (poss_df['pointsScored'] > 0).astype(int)

    def calculate_streak(series):
        """Calculate current streak (consecutive 1s or 0s)"""
        streak = 0
        result = []
        for val in series:
            if val == 1:
                streak = max(1, streak + 1)
            else:
                streak = min(-1, streak - 1)
            result.append(streak)
        return result

    poss_df['team_shooting_streak'] = (
        poss_df.groupby('gameId')['scored_possession']
        .transform(calculate_streak)
    )

    # Team rolling efficiency (last 10 possessions)
    poss_df['team_rolling_ppg'] = (
        poss_df.groupby('gameId')['pointsScored']
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Opponent rolling defense (points allowed last 10)
    poss_df['opp_rolling_defense'] = (
        poss_df.groupby('gameId')['pointsAllowed']
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Feature 4: Game context features
    print("\nAdding game context features...")

    # Time pressure (late game + close score)
    poss_df['time_pressure'] = (
        (poss_df['period'] >= 4) &
        (np.abs(poss_df['scoreDiffStart']) <= 5) &
        (poss_df['clockStart'] < 120)  # Last 2 minutes
    ).astype(int)

    # Comeback momentum (negative score diff + recent scoring)
    poss_df['comeback_momentum'] = (
        (poss_df['scoreDiffStart'] < -5) &
        (poss_df['pointsLast3'] >= 6)
    ).astype(int)

    # Feature 5: Aggregate player stats from shooting events
    print("\nAggregating player statistics by game...")

    # Calculate average player stats per game
    game_player_stats = shooting_events.groupby('gameId').agg({
        'player_rolling_fg_pct': 'mean',
        'player_hot_hand': 'mean',
        'player_recent_makes': 'mean',
    }).reset_index()

    game_player_stats.columns = ['gameId', 'avg_player_fg_pct', 'avg_hot_hand_pct', 'avg_recent_makes']

    # Merge back to possessions
    poss_df = poss_df.merge(game_player_stats, on='gameId', how='left')

    # Fill NaN values
    poss_df['avg_player_fg_pct'] = poss_df['avg_player_fg_pct'].fillna(0.45)  # League average
    poss_df['avg_hot_hand_pct'] = poss_df['avg_hot_hand_pct'].fillna(0.5)
    poss_df['avg_recent_makes'] = poss_df['avg_recent_makes'].fillna(2.5)

    print("✓ All features added")

    # Summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)

    new_features = [
        'team_shooting_streak',
        'team_rolling_ppg',
        'opp_rolling_defense',
        'time_pressure',
        'comeback_momentum',
        'avg_player_fg_pct',
        'avg_hot_hand_pct',
        'avg_recent_makes',
    ]

    print(f"\nAdded {len(new_features)} new features:")
    for feat in new_features:
        mean_val = poss_df[feat].mean()
        std_val = poss_df[feat].std()
        print(f"  - {feat:25s} mean={mean_val:6.3f}, std={std_val:6.3f}")

    # Save
    poss_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved enhanced possessions to {output_file}")
    print(f"  Total features: {len(poss_df.columns)}")

    return poss_df, new_features


if __name__ == "__main__":
    poss_df, new_features = add_player_rolling_stats(
        'possessions_multi_season.csv',
        'possessions_multi_season_with_players.csv'
    )

    print("\n" + "="*60)
    print("NEXT STEP")
    print("="*60)
    print("""
    Now retrain HMM with these new features:

    python train_hmm_with_players.py

    Expected improvements:
    - AUC: 0.49 → 0.52-0.55
    - Better state separation
    - More meaningful momentum states
    """)
