"""
Feature Optimization - Find the most important features

Tries:
1. Feature importance analysis
2. Feature selection
3. Correlation analysis
4. Dimensionality reduction
"""
import sys
sys.path.append('/Users/xavier/Downloads/NBAMomentumHMM-main')

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from train_hmm_enhanced import create_enhanced_features, MomentumHMM


def analyze_feature_importance(df, states, output_prefix='feature_analysis'):
    """Analyze which features are most important for prediction"""

    print("="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    df_eval = df.copy()
    df_eval['state'] = states

    # Create target
    df_eval['y_scored'] = (df_eval['pointsScored'] > 0).astype(int)
    df_eval['y_next'] = df_eval['y_scored'].shift(-1)

    # Filter
    offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
    df_eval = df_eval[df_eval["result"].isin(offense_results)]
    df_eval = df_eval.dropna(subset=['y_next'])
    df_eval['next_gameId'] = df_eval['gameId'].shift(-1)
    df_eval = df_eval[df_eval['gameId'] == df_eval['next_gameId']]

    # All features
    all_features = [
        'scoreDiffStart', 'pointsScored', 'pointsAllowed', 'shotDistance',
        'foulsCommitted', 'foulsDrawn', 'oRebsGrabbed', 'runMagnitude',
        'offensiveEfficiency', 'momentumScore', 'gameImportance', 'periodPressure',
        'secondChance', 'foulAdvantage', 'recentScoringRate',
        'team_shooting_streak', 'team_rolling_ppg', 'opp_rolling_defense',
        'time_pressure', 'comeback_momentum', 'avg_player_fg_pct',
        'avg_hot_hand_pct', 'avg_recent_makes',
    ]

    X = df_eval[all_features].fillna(0)
    y = df_eval['y_next']

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nDataset size: {len(X_train):,} train, {len(X_test):,} test")

    # ===== Method 1: Logistic Regression Coefficients =====
    print("\n" + "="*60)
    print("METHOD 1: Logistic Regression Feature Importance")
    print("="*60)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    # Get coefficients
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'coefficient': lr.coef_[0],
        'abs_coefficient': np.abs(lr.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    print("\nTop 10 Most Important Features (by coefficient):")
    print(feature_importance.head(10).to_string(index=False))

    # ===== Method 2: Mutual Information =====
    print("\n" + "="*60)
    print("METHOD 2: Mutual Information")
    print("="*60)

    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': all_features,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    print("\nTop 10 Features by Mutual Information:")
    print(mi_importance.head(10).to_string(index=False))

    # ===== Method 3: F-statistic (ANOVA) =====
    print("\n" + "="*60)
    print("METHOD 3: ANOVA F-statistic")
    print("="*60)

    f_scores, p_values = f_classif(X_train, y_train)
    f_importance = pd.DataFrame({
        'feature': all_features,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)

    print("\nTop 10 Features by F-statistic:")
    print(f_importance.head(10).to_string(index=False))

    # ===== Method 4: Correlation Analysis =====
    print("\n" + "="*60)
    print("METHOD 4: Feature Correlation Matrix")
    print("="*60)

    correlation_matrix = X_train.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': correlation_matrix.iloc[i, j]
                })

    if high_corr_pairs:
        print("\nHighly Correlated Features (|r| > 0.8):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']:25s} <-> {pair['feature2']:25s} : {pair['correlation']:.3f}")
    else:
        print("\nNo highly correlated features found (|r| > 0.8)")

    # ===== Test Feature Selection =====
    print("\n" + "="*60)
    print("TESTING FEATURE SELECTION")
    print("="*60)

    results = []

    # Baseline (all features)
    lr_full = LogisticRegression(max_iter=1000, random_state=42)
    lr_full.fit(X_train_scaled, y_train)
    y_prob_full = lr_full.predict_proba(X_test_scaled)[:, 1]
    auc_full = roc_auc_score(y_test, y_prob_full)
    results.append(('All 23 features', 23, auc_full))
    print(f"\nBaseline (all features): AUC = {auc_full:.4f}")

    # Test different numbers of features
    for k in [5, 10, 15, 20]:
        # Select top K by mutual information
        selector = SelectKBest(mutual_info_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        scaler_k = StandardScaler()
        X_train_selected_scaled = scaler_k.fit_transform(X_train_selected)
        X_test_selected_scaled = scaler_k.transform(X_test_selected)

        lr_k = LogisticRegression(max_iter=1000, random_state=42)
        lr_k.fit(X_train_selected_scaled, y_train)
        y_prob_k = lr_k.predict_proba(X_test_selected_scaled)[:, 1]
        auc_k = roc_auc_score(y_test, y_prob_k)

        results.append((f'Top {k} features', k, auc_k))
        print(f"Top {k:2d} features (MI): AUC = {auc_k:.4f} ({auc_k - auc_full:+.4f})")

        # Show selected features
        selected_features = [all_features[i] for i in selector.get_support(indices=True)]
        print(f"  Selected: {', '.join(selected_features[:5])}{'...' if k > 5 else ''}")

    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)

    # Find best configuration
    best_config = max(results, key=lambda x: x[2])
    print(f"\nBest Configuration: {best_config[0]} (AUC: {best_config[2]:.4f})")

    # Consensus top features (appear in top 10 of at least 2 methods)
    top_lr = set(feature_importance.head(10)['feature'])
    top_mi = set(mi_importance.head(10)['feature'])
    top_f = set(f_importance.head(10)['feature'])

    consensus_features = []
    for feat in all_features:
        score = (feat in top_lr) + (feat in top_mi) + (feat in top_f)
        if score >= 2:
            consensus_features.append((feat, score))

    consensus_features.sort(key=lambda x: x[1], reverse=True)

    print("\n🌟 Consensus Top Features (in ≥2 methods' top 10):")
    for feat, score in consensus_features:
        print(f"  {'⭐' * score} {feat}")

    # Save results
    with open(f'{output_prefix}_importance.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write("Top 10 by Logistic Regression:\n")
        f.write(feature_importance.head(10).to_string(index=False) + "\n\n")

        f.write("Top 10 by Mutual Information:\n")
        f.write(mi_importance.head(10).to_string(index=False) + "\n\n")

        f.write("Top 10 by F-statistic:\n")
        f.write(f_importance.head(10).to_string(index=False) + "\n\n")

        f.write("Consensus Top Features:\n")
        for feat, score in consensus_features:
            f.write(f"  {'⭐' * score} {feat}\n")

    print(f"\n✓ Detailed results saved to {output_prefix}_importance.txt")

    return {
        'lr_importance': feature_importance,
        'mi_importance': mi_importance,
        'f_importance': f_importance,
        'consensus': consensus_features,
        'selection_results': results,
    }


def test_optimized_hmm(df, selected_features, n_states=4):
    """Test HMM with optimized feature set"""

    print("\n" + "="*60)
    print(f"TRAINING HMM WITH {len(selected_features)} OPTIMIZED FEATURES")
    print("="*60)
    print(f"Features: {', '.join(selected_features)}")

    hmm = MomentumHMM(n_states=n_states, n_iter=100, random_state=42)
    X, lengths = hmm.prepare_sequences(df, selected_features)
    hmm.fit(X, lengths)
    states = hmm.predict_states(X, lengths)

    # Evaluate
    df_eval = df.copy()
    df_eval['state'] = states
    df_eval['y_scored'] = (df_eval['pointsScored'] > 0).astype(int)
    df_eval['y_next'] = df_eval['y_scored'].shift(-1)

    offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
    df_eval = df_eval[df_eval["result"].isin(offense_results)]
    df_eval = df_eval.dropna(subset=['y_next'])
    df_eval['next_gameId'] = df_eval['gameId'].shift(-1)
    df_eval = df_eval[df_eval['gameId'] == df_eval['next_gameId']]

    # Features + states
    base_features = ["scoreDiffStart", "period", "clockStart",
                    "foulsCommitted", "foulsDrawn", "oRebsGrabbed"]

    state_dummies = pd.get_dummies(df_eval['state'], prefix='momentum_state')
    df_eval = pd.concat([df_eval, state_dummies], axis=1)
    momentum_features = state_dummies.columns.tolist()

    split_idx = int(len(df_eval) * 0.8)
    train_df = df_eval.iloc[:split_idx]
    test_df = df_eval.iloc[split_idx:]

    # Train
    X_train = train_df[base_features + momentum_features].fillna(0)
    y_train = train_df['y_next']
    X_test = test_df[base_features + momentum_features].fillna(0)
    y_test = test_df['y_next']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nOptimized HMM AUC: {auc:.4f}")

    return auc, hmm, states


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('possessions_multi_season_with_players.csv')
    df_enhanced = create_enhanced_features(df)

    all_features = [
        'scoreDiffStart', 'pointsScored', 'pointsAllowed', 'shotDistance',
        'foulsCommitted', 'foulsDrawn', 'oRebsGrabbed', 'runMagnitude',
        'offensiveEfficiency', 'momentumScore', 'gameImportance', 'periodPressure',
        'secondChance', 'foulAdvantage', 'recentScoringRate',
        'team_shooting_streak', 'team_rolling_ppg', 'opp_rolling_defense',
        'time_pressure', 'comeback_momentum', 'avg_player_fg_pct',
        'avg_hot_hand_pct', 'avg_recent_makes',
    ]

    # First, train HMM with all features to get states
    print("\nTraining initial HMM (for state discovery)...")
    hmm_full = MomentumHMM(n_states=4, n_iter=100, random_state=42)
    X, lengths = hmm_full.prepare_sequences(df_enhanced, all_features)
    hmm_full.fit(X, lengths)
    states = hmm_full.predict_states(X, lengths)

    # Analyze features
    results = analyze_feature_importance(df_enhanced, states)

    # Test with consensus features
    if results['consensus']:
        consensus_features = [feat for feat, _ in results['consensus']]
        print(f"\n{'='*60}")
        print("TESTING CONSENSUS FEATURES")
        print(f"{'='*60}")
        test_optimized_hmm(df_enhanced, consensus_features, n_states=4)
