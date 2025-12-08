"""
Enhanced HMM Training with Better Feature Engineering
Goal: Achieve more balanced state distribution while keeping 3 states
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/xavier/Downloads/NBAMomentumHMM-main')
from train_hmm import MomentumHMM, evaluate_momentum_prediction, print_results
import argparse

def create_enhanced_features(df):
    """
    Create enhanced features to help HMM discover better state separation
    """
    df_enhanced = df.copy()

    # 1. Offensive efficiency (normalized by shot distance)
    df_enhanced['offensiveEfficiency'] = df_enhanced['pointsScored'] / (1 + df_enhanced['shotDistance'] / 10)
    df_enhanced['offensiveEfficiency'] = df_enhanced['offensiveEfficiency'].fillna(0)

    # 2. Momentum indicator (recent performance + current)
    df_enhanced['momentumScore'] = (
        df_enhanced['pointsScored'] * 2 +  # Current possession weighted more
        df_enhanced['pointsLast3'] +
        df_enhanced['runMagnitude'] / 5 -
        df_enhanced['pointsAllowed'] * 1.5
    )

    # 3. Game context (closer games = higher importance)
    df_enhanced['gameImportance'] = 1 / (1 + np.abs(df_enhanced['scoreDiffStart']) / 10)

    # 4. Period pressure (4th quarter = higher pressure)
    df_enhanced['periodPressure'] = df_enhanced['period'] * (1 - df_enhanced['clockStart'] / 720)

    # 5. Second chance opportunity
    df_enhanced['secondChance'] = df_enhanced['oRebsGrabbed'] - df_enhanced['oRebsAllowed']

    # 6. Foul advantage
    df_enhanced['foulAdvantage'] = df_enhanced['foulsDrawn'] - df_enhanced['foulsCommitted']

    # 7. Scoring rate (points per possession in recent stretch)
    df_enhanced['recentScoringRate'] = df_enhanced['pointsLast5'] / 5.0

    return df_enhanced


def train_enhanced_hmm(data_file='possessions_multi_season.csv', n_states=3, n_iter=100):
    """Train HMM with enhanced features"""

    print("="*60)
    print("ENHANCED HMM TRAINING")
    print("Goal: Better state separation with 3 states")
    print("="*60)

    # Load data
    df = pd.read_csv(data_file)
    print(f"\nLoaded {len(df):,} possessions from {df['gameId'].nunique()} games")

    # Create enhanced features
    print("\nCreating enhanced features...")
    df_enhanced = create_enhanced_features(df)

    # Enhanced feature set for HMM
    enhanced_features = [
        # Original features
        'scoreDiffStart',
        'pointsScored',
        'pointsAllowed',
        'shotDistance',
        'foulsCommitted',
        'foulsDrawn',
        'oRebsGrabbed',
        'runMagnitude',
        # New composite features
        'offensiveEfficiency',
        'momentumScore',
        'gameImportance',
        'periodPressure',
        'secondChance',
        'foulAdvantage',
        'recentScoringRate',
    ]

    print(f"Using {len(enhanced_features)} features:")
    for feat in enhanced_features:
        print(f"  - {feat}")

    # Train HMM
    print(f"\nTraining HMM with {n_states} states...")
    momentum_hmm = MomentumHMM(n_states=n_states, n_iter=n_iter, random_state=42)

    X, lengths = momentum_hmm.prepare_sequences(df_enhanced, enhanced_features)
    momentum_hmm.fit(X, lengths)

    # Predict states
    states = momentum_hmm.predict_states(X, lengths)

    # Analyze states
    state_stats, state_order = momentum_hmm.analyze_states(df_enhanced, states)
    transition_matrix = momentum_hmm.get_transition_matrix()

    # Print state distribution
    print("\n" + "="*60)
    print("STATE DISTRIBUTION")
    print("="*60)
    unique, counts = np.unique(states, return_counts=True)
    for state, count in zip(unique, counts):
        pct = count / len(states) * 100
        label = momentum_hmm.state_labels.get(state, f'State {state}')
        print(f"{label:12s} (State {state}): {count:6,} samples ({pct:5.1f}%)")

    # Detailed state analysis
    print("\n" + "="*60)
    print("DETAILED STATE CHARACTERISTICS")
    print("="*60)

    df_enhanced['state'] = states
    for state_num, label in momentum_hmm.state_labels.items():
        state_data = df_enhanced[df_enhanced['state'] == state_num]
        print(f"\n{label} (State {state_num}):")
        print(f"  Sample count: {len(state_data):,}")
        print(f"  Avg points scored: {state_data['pointsScored'].mean():.3f}")
        print(f"  Avg offensive efficiency: {state_data['offensiveEfficiency'].mean():.3f}")
        print(f"  Avg momentum score: {state_data['momentumScore'].mean():.3f}")
        print(f"  Avg shot distance: {state_data['shotDistance'].mean():.2f} ft")
        print(f"  Avg game importance: {state_data['gameImportance'].mean():.3f}")
        print(f"  % with points: {(state_data['pointsScored'] > 0).sum() / len(state_data) * 100:.1f}%")

    # Evaluate prediction (use original df for compatibility)
    print("\n" + "="*60)
    print("PREDICTION EVALUATION")
    print("="*60)

    results, model, test_data = evaluate_momentum_prediction(df, states, momentum_hmm)

    # Print results
    print_results(results, state_stats, transition_matrix, momentum_hmm.state_labels)

    # Save model
    import pickle
    with open('momentum_hmm_enhanced.pkl', 'wb') as f:
        pickle.dump((momentum_hmm, enhanced_features), f)
    print("\n✓ Enhanced model saved to momentum_hmm_enhanced.pkl")

    return momentum_hmm, df_enhanced, states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='possessions_multi_season.csv')
    parser.add_argument('--n_states', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=100)
    args = parser.parse_args()

    train_enhanced_hmm(args.data, args.n_states, args.n_iter)
