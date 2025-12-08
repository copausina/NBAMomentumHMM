"""
Class Imbalance Optimization

Tests different sampling strategies:
1. SMOTE (Synthetic Minority Over-sampling)
2. Random oversampling
3. Random undersampling
4. Class weights
"""
import sys
sys.path.append('/Users/xavier/Downloads/NBAMomentumHMM-main')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from train_hmm_enhanced import create_enhanced_features, MomentumHMM


def test_sampling_strategies(df, states):
    """Test different sampling strategies"""

    print("="*60)
    print("CLASS IMBALANCE OPTIMIZATION")
    print("="*60)

    df_eval = df.copy()
    df_eval['state'] = states
    df_eval['y_scored'] = (df_eval['pointsScored'] > 0).astype(int)
    df_eval['y_next'] = df_eval['y_scored'].shift(-1)

    offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
    df_eval = df_eval[df_eval["result"].isin(offense_results)]
    df_eval = df_eval.dropna(subset=['y_next'])
    df_eval['next_gameId'] = df_eval['gameId'].shift(-1)
    df_eval = df_eval[df_eval['gameId'] == df_eval['next_gameId']]

    # Features
    base_features = ["scoreDiffStart", "period", "clockStart",
                    "foulsCommitted", "foulsDrawn", "oRebsGrabbed"]
    state_dummies = pd.get_dummies(df_eval['state'], prefix='momentum_state')
    df_eval = pd.concat([df_eval, state_dummies], axis=1)
    momentum_features = state_dummies.columns.tolist()
    all_features = base_features + momentum_features

    # Split
    split_idx = int(len(df_eval) * 0.8)
    train_df = df_eval.iloc[:split_idx]
    test_df = df_eval.iloc[split_idx:]

    X_train = train_df[all_features].fillna(0).values
    y_train = train_df['y_next'].values
    X_test = test_df[all_features].fillna(0).values
    y_test = test_df['y_next'].values

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nOriginal class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:,} ({count/len(y_train)*100:.1f}%)")

    results = []

    # ===== Baseline (no sampling) =====
    print("\n" + "="*60)
    print("BASELINE (No Sampling)")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc_baseline = roc_auc_score(y_test, y_prob)

    print(f"AUC: {auc_baseline:.4f}")
    results.append(('Baseline', auc_baseline))

    # ===== SMOTE =====
    print("\n" + "="*60)
    print("SMOTE (Synthetic Minority Oversampling)")
    print("="*60)

    try:
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        print(f"After SMOTE:")
        unique, counts = np.unique(y_train_smote, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count:,} ({count/len(y_train_smote)*100:.1f}%)")

        scaler_smote = StandardScaler()
        X_train_smote_scaled = scaler_smote.fit_transform(X_train_smote)
        X_test_smote_scaled = scaler_smote.transform(X_test)

        model_smote = LogisticRegression(max_iter=1000, random_state=42)
        model_smote.fit(X_train_smote_scaled, y_train_smote)
        y_prob_smote = model_smote.predict_proba(X_test_smote_scaled)[:, 1]
        auc_smote = roc_auc_score(y_test, y_prob_smote)

        print(f"AUC: {auc_smote:.4f} ({auc_smote - auc_baseline:+.4f})")
        results.append(('SMOTE', auc_smote))
    except Exception as e:
        print(f"SMOTE failed: {e}")

    # ===== Random Oversampling =====
    print("\n" + "="*60)
    print("Random Oversampling")
    print("="*60)

    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    scaler_ros = StandardScaler()
    X_train_ros_scaled = scaler_ros.fit_transform(X_train_ros)
    X_test_ros_scaled = scaler_ros.transform(X_test)

    model_ros = LogisticRegression(max_iter=1000, random_state=42)
    model_ros.fit(X_train_ros_scaled, y_train_ros)
    y_prob_ros = model_ros.predict_proba(X_test_ros_scaled)[:, 1]
    auc_ros = roc_auc_score(y_test, y_prob_ros)

    print(f"AUC: {auc_ros:.4f} ({auc_ros - auc_baseline:+.4f})")
    results.append(('Random Oversample', auc_ros))

    # ===== Random Undersampling =====
    print("\n" + "="*60)
    print("Random Undersampling")
    print("="*60)

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    scaler_rus = StandardScaler()
    X_train_rus_scaled = scaler_rus.fit_transform(X_train_rus)
    X_test_rus_scaled = scaler_rus.transform(X_test)

    model_rus = LogisticRegression(max_iter=1000, random_state=42)
    model_rus.fit(X_train_rus_scaled, y_train_rus)
    y_prob_rus = model_rus.predict_proba(X_test_rus_scaled)[:, 1]
    auc_rus = roc_auc_score(y_test, y_prob_rus)

    print(f"AUC: {auc_rus:.4f} ({auc_rus - auc_baseline:+.4f})")
    results.append(('Random Undersample', auc_rus))

    # ===== Class Weights =====
    print("\n" + "="*60)
    print("Class Weights (balanced)")
    print("="*60)

    model_weighted = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model_weighted.fit(X_train_scaled, y_train)
    y_prob_weighted = model_weighted.predict_proba(X_test_scaled)[:, 1]
    auc_weighted = roc_auc_score(y_test, y_prob_weighted)

    print(f"AUC: {auc_weighted:.4f} ({auc_weighted - auc_baseline:+.4f})")
    results.append(('Class Weights', auc_weighted))

    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    results_df = pd.DataFrame(results, columns=['Method', 'AUC'])
    results_df['Improvement'] = results_df['AUC'] - auc_baseline
    results_df = results_df.sort_values('AUC', ascending=False)

    print("\n" + results_df.to_string(index=False))

    best_method = results_df.iloc[0]
    print(f"\n🏆 Best Method: {best_method['Method']} (AUC: {best_method['AUC']:.4f}, +{best_method['Improvement']:.4f})")

    return results_df


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('possessions_multi_season_with_players.csv')
    df_enhanced = create_enhanced_features(df)

    # Train HMM to get states
    print("\nTraining HMM (4 states)...")
    all_features = [
        'scoreDiffStart', 'pointsScored', 'pointsAllowed', 'shotDistance',
        'foulsCommitted', 'foulsDrawn', 'oRebsGrabbed', 'runMagnitude',
        'offensiveEfficiency', 'momentumScore', 'gameImportance', 'periodPressure',
        'secondChance', 'foulAdvantage', 'recentScoringRate',
        'team_shooting_streak', 'team_rolling_ppg', 'opp_rolling_defense',
        'time_pressure', 'comeback_momentum', 'avg_player_fg_pct',
        'avg_hot_hand_pct', 'avg_recent_makes',
    ]

    hmm = MomentumHMM(n_states=4, n_iter=100, random_state=42)
    X, lengths = hmm.prepare_sequences(df_enhanced, all_features)
    hmm.fit(X, lengths)
    states = hmm.predict_states(X, lengths)

    # Test sampling strategies
    results = test_sampling_strategies(df_enhanced, states)
