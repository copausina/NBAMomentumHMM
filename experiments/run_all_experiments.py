"""
Comprehensive Experiments - All Prediction Targets & Configurations

Tests:
1. Different prediction targets (next-1, next-3, next-5, regression)
2. Different number of states (2, 3, 4, 5)
3. Saves all results to JSON for comparison
"""
import sys
sys.path.append('/Users/xavier/Downloads/NBAMomentumHMM-main')

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, r2_score
from train_hmm_enhanced import create_enhanced_features, MomentumHMM


def evaluate_multi_possession_prediction(df, states, hmm_model, n_ahead=1, task='classification'):
    """
    Evaluate prediction for next N possessions

    Args:
        n_ahead: Number of possessions to predict ahead
        task: 'classification' (scored or not) or 'regression' (total points)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {n_ahead}-POSSESSION-AHEAD {'CLASSIFICATION' if task == 'classification' else 'REGRESSION'}")
    print(f"{'='*60}")

    df_eval = df.copy()
    df_eval['state'] = states

    # Create prediction targets
    if task == 'classification':
        # Binary: will score in next N possessions?
        if n_ahead == 1:
            df_eval['y_target'] = (df_eval['pointsScored'] > 0).astype(int)
            df_eval['y_next'] = df_eval['y_target'].shift(-1)
        else:
            # Did team score in ANY of next N possessions?
            df_eval['points_next_n'] = sum(df_eval['pointsScored'].shift(-i) for i in range(1, n_ahead+1))
            df_eval['y_next'] = (df_eval['points_next_n'] > 0).astype(int)
    else:  # regression
        # Total points in next N possessions
        df_eval['y_next'] = sum(df_eval['pointsScored'].shift(-i) for i in range(1, n_ahead+1))

    # Filter valid possessions
    offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
    df_eval = df_eval[df_eval["result"].isin(offense_results)]

    # Remove last N possessions of each game
    df_eval = df_eval.dropna(subset=['y_next'])
    for i in range(1, n_ahead+1):
        df_eval[f'next_gameId_{i}'] = df_eval['gameId'].shift(-i)

    # Keep only if all next N possessions are in same game
    mask = True
    for i in range(1, n_ahead+1):
        mask = mask & (df_eval['gameId'] == df_eval[f'next_gameId_{i}'])
    df_eval = df_eval[mask]

    if len(df_eval) == 0:
        print(f"  ⚠ No valid samples for {n_ahead}-ahead prediction")
        return None

    print(f"\nEvaluation dataset: {len(df_eval):,} possessions")

    # Features
    base_features = [
        "scoreDiffStart", "period", "clockStart",
        "foulsCommitted", "foulsDrawn", "oRebsGrabbed"
    ]

    # One-hot encode states
    state_dummies = pd.get_dummies(df_eval['state'], prefix='momentum_state')
    df_eval = pd.concat([df_eval, state_dummies], axis=1)
    momentum_features = state_dummies.columns.tolist()

    # Split by time
    split_idx = int(len(df_eval) * 0.8)
    train_df = df_eval.iloc[:split_idx]
    test_df = df_eval.iloc[split_idx:]

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Baseline model
    print(f"\n[1/2] Training baseline model...")
    X_train_base = train_df[base_features].fillna(0)
    y_train = train_df['y_next']
    X_test_base = test_df[base_features].fillna(0)
    y_test = test_df['y_next']

    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)

    if task == 'classification':
        model_base = LogisticRegression(max_iter=1000, random_state=42)
        model_base.fit(X_train_base_scaled, y_train)
        y_pred_base = model_base.predict(X_test_base_scaled)
        y_prob_base = model_base.predict_proba(X_test_base_scaled)[:, 1]
    else:  # regression
        model_base = Ridge(random_state=42)
        model_base.fit(X_train_base_scaled, y_train)
        y_pred_base = model_base.predict(X_test_base_scaled)

    # Momentum model
    print(f"[2/2] Training model WITH momentum states...")
    X_train_mom = train_df[base_features + momentum_features].fillna(0)
    X_test_mom = test_df[base_features + momentum_features].fillna(0)

    scaler_mom = StandardScaler()
    X_train_mom_scaled = scaler_mom.fit_transform(X_train_mom)
    X_test_mom_scaled = scaler_mom.transform(X_test_mom)

    if task == 'classification':
        model_mom = LogisticRegression(max_iter=1000, random_state=42)
        model_mom.fit(X_train_mom_scaled, y_train)
        y_pred_mom = model_mom.predict(X_test_mom_scaled)
        y_prob_mom = model_mom.predict_proba(X_test_mom_scaled)[:, 1]
    else:  # regression
        model_mom = Ridge(random_state=42)
        model_mom.fit(X_train_mom_scaled, y_train)
        y_pred_mom = model_mom.predict(X_test_mom_scaled)

    # Compute metrics
    if task == 'classification':
        results = {
            'baseline': {
                'accuracy': accuracy_score(y_test, y_pred_base),
                'auc': roc_auc_score(y_test, y_prob_base),
                'log_loss': log_loss(y_test, y_prob_base),
            },
            'with_momentum': {
                'accuracy': accuracy_score(y_test, y_pred_mom),
                'auc': roc_auc_score(y_test, y_prob_mom),
                'log_loss': log_loss(y_test, y_prob_mom),
            }
        }
        results['improvement'] = {
            'accuracy': results['with_momentum']['accuracy'] - results['baseline']['accuracy'],
            'auc': results['with_momentum']['auc'] - results['baseline']['auc'],
            'log_loss_reduction': results['baseline']['log_loss'] - results['with_momentum']['log_loss'],
        }
    else:  # regression
        results = {
            'baseline': {
                'mse': mean_squared_error(y_test, y_pred_base),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_base)),
                'r2': r2_score(y_test, y_pred_base),
            },
            'with_momentum': {
                'mse': mean_squared_error(y_test, y_pred_mom),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_mom)),
                'r2': r2_score(y_test, y_pred_mom),
            }
        }
        results['improvement'] = {
            'mse_reduction': results['baseline']['mse'] - results['with_momentum']['mse'],
            'rmse_reduction': results['baseline']['rmse'] - results['with_momentum']['rmse'],
            'r2_improvement': results['with_momentum']['r2'] - results['baseline']['r2'],
        }

    return results


def train_hmm_n_states(df, n_states, features):
    """Train HMM with specified number of states"""
    print(f"\n{'='*60}")
    print(f"TRAINING HMM WITH {n_states} STATES")
    print(f"{'='*60}")

    momentum_hmm = MomentumHMM(n_states=n_states, n_iter=100, random_state=42)
    X, lengths = momentum_hmm.prepare_sequences(df, features)
    momentum_hmm.fit(X, lengths)
    states = momentum_hmm.predict_states(X, lengths)

    # State distribution
    unique, counts = np.unique(states, return_counts=True)
    state_dist = {int(s): {'count': int(c), 'pct': float(c/len(states)*100)}
                  for s, c in zip(unique, counts)}

    return momentum_hmm, states, state_dist


def run_all_experiments():
    """Run comprehensive experiments"""

    print("="*60)
    print("NBA MOMENTUM HMM - COMPREHENSIVE EXPERIMENTS")
    print("="*60)
    print(f"Start time: {datetime.now()}")

    # Load data with player features
    df = pd.read_csv('possessions_multi_season_with_players.csv')
    print(f"\nLoaded {len(df):,} possessions from {df['gameId'].nunique()} games")

    # Create enhanced features
    df_enhanced = create_enhanced_features(df)

    # Feature set
    all_features = [
        'scoreDiffStart', 'pointsScored', 'pointsAllowed', 'shotDistance',
        'foulsCommitted', 'foulsDrawn', 'oRebsGrabbed', 'runMagnitude',
        'offensiveEfficiency', 'momentumScore', 'gameImportance', 'periodPressure',
        'secondChance', 'foulAdvantage', 'recentScoringRate',
        'team_shooting_streak', 'team_rolling_ppg', 'opp_rolling_defense',
        'time_pressure', 'comeback_momentum', 'avg_player_fg_pct',
        'avg_hot_hand_pct', 'avg_recent_makes',
    ]

    all_results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'n_possessions': len(df),
            'n_games': df['gameId'].nunique(),
            'n_features': len(all_features),
        },
        'experiments': {}
    }

    # ===== EXPERIMENT 1: Different Prediction Targets (with n_states=3) =====
    print("\n" + "="*60)
    print("EXPERIMENT 1: DIFFERENT PREDICTION TARGETS")
    print("="*60)

    # Train HMM with 3 states once
    hmm_3states, states_3, state_dist_3 = train_hmm_n_states(df_enhanced, 3, all_features)

    prediction_targets = [
        ('next_1_class', 1, 'classification'),
        ('next_3_class', 3, 'classification'),
        ('next_5_class', 5, 'classification'),
        ('next_1_regr', 1, 'regression'),
        ('next_3_regr', 3, 'regression'),
        ('next_5_regr', 5, 'regression'),
    ]

    for exp_name, n_ahead, task in prediction_targets:
        print(f"\n--- Testing: {exp_name} ---")
        results = evaluate_multi_possession_prediction(df, states_3, hmm_3states, n_ahead, task)

        if results:
            all_results['experiments'][exp_name] = {
                'n_states': 3,
                'n_ahead': n_ahead,
                'task': task,
                'state_distribution': state_dist_3,
                'results': results,
            }

            # Print summary
            print(f"\n{'='*60}")
            print(f"RESULTS: {exp_name}")
            print(f"{'='*60}")
            if task == 'classification':
                print(f"Baseline AUC: {results['baseline']['auc']:.4f}")
                print(f"Momentum AUC: {results['with_momentum']['auc']:.4f}")
                print(f"Improvement:  {results['improvement']['auc']:+.4f}")
            else:
                print(f"Baseline RMSE: {results['baseline']['rmse']:.4f}")
                print(f"Momentum RMSE: {results['with_momentum']['rmse']:.4f}")
                print(f"Improvement:   {results['improvement']['rmse_reduction']:+.4f}")
                print(f"Baseline R²:   {results['baseline']['r2']:.4f}")
                print(f"Momentum R²:   {results['with_momentum']['r2']:.4f}")

    # ===== EXPERIMENT 2: Different Number of States (with next-1 classification) =====
    print("\n" + "="*60)
    print("EXPERIMENT 2: DIFFERENT NUMBER OF STATES")
    print("="*60)

    for n_states in [2, 4, 5]:
        exp_name = f'n_states_{n_states}'
        print(f"\n--- Testing: {n_states} states ---")

        hmm, states, state_dist = train_hmm_n_states(df_enhanced, n_states, all_features)
        results = evaluate_multi_possession_prediction(df, states, hmm, n_ahead=1, task='classification')

        if results:
            all_results['experiments'][exp_name] = {
                'n_states': n_states,
                'n_ahead': 1,
                'task': 'classification',
                'state_distribution': state_dist,
                'results': results,
            }

            print(f"\n{'='*60}")
            print(f"RESULTS: {n_states} states")
            print(f"{'='*60}")
            print(f"Baseline AUC: {results['baseline']['auc']:.4f}")
            print(f"Momentum AUC: {results['with_momentum']['auc']:.4f}")
            print(f"Improvement:  {results['improvement']['auc']:+.4f}")
            print(f"\nState distribution:")
            for s, info in state_dist.items():
                print(f"  State {s}: {info['count']:,} ({info['pct']:.1f}%)")

    # Save all results
    output_file = 'experiment_results_all.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"End time: {datetime.now()}")
    print(f"Results saved to: {output_file}")

    # Generate summary report
    generate_summary_report(all_results)

    return all_results


def generate_summary_report(results):
    """Generate a markdown summary report"""

    report = f"""# Comprehensive Experiment Results

**Date**: {results['metadata']['date']}
**Dataset**: {results['metadata']['n_possessions']:,} possessions from {results['metadata']['n_games']} games

---

## 🎯 EXPERIMENT 1: Different Prediction Targets (n_states=3)

| Target | Task | Baseline AUC/RMSE | Momentum AUC/RMSE | Improvement |
|--------|------|-------------------|-------------------|-------------|
"""

    for exp_name in ['next_1_class', 'next_3_class', 'next_5_class', 'next_1_regr', 'next_3_regr', 'next_5_regr']:
        if exp_name in results['experiments']:
            exp = results['experiments'][exp_name]
            task = exp['task']
            if task == 'classification':
                base_metric = exp['results']['baseline']['auc']
                mom_metric = exp['results']['with_momentum']['auc']
                improvement = exp['results']['improvement']['auc']
                metric_name = "AUC"
            else:
                base_metric = exp['results']['baseline']['rmse']
                mom_metric = exp['results']['with_momentum']['rmse']
                improvement = exp['results']['improvement']['rmse_reduction']
                metric_name = "RMSE"

            report += f"| {exp_name} | {task[:5]} | {base_metric:.4f} | {mom_metric:.4f} | {improvement:+.4f} |\n"

    report += f"""
---

## 🔢 EXPERIMENT 2: Different Number of States (next-1 classification)

| States | Baseline AUC | Momentum AUC | Improvement | State Distribution |
|--------|--------------|--------------|-------------|-------------------|
"""

    for n_states in [2, 3, 4, 5]:
        exp_name = f'n_states_{n_states}' if n_states != 3 else 'next_1_class'
        if exp_name in results['experiments']:
            exp = results['experiments'][exp_name]
            base_auc = exp['results']['baseline']['auc']
            mom_auc = exp['results']['with_momentum']['auc']
            improvement = exp['results']['improvement']['auc']

            dist_str = ", ".join([f"S{s}:{info['pct']:.1f}%"
                                 for s, info in sorted(exp['state_distribution'].items())])

            report += f"| {n_states} | {base_auc:.4f} | {mom_auc:.4f} | {improvement:+.4f} | {dist_str} |\n"

    report += """
---

## 📊 Key Findings

### Best Performance
"""

    # Find best AUC
    best_auc = 0
    best_exp = None
    for exp_name, exp in results['experiments'].items():
        if exp['task'] == 'classification':
            auc = exp['results']['with_momentum']['auc']
            if auc > best_auc:
                best_auc = auc
                best_exp = exp_name

    if best_exp:
        exp = results['experiments'][best_exp]
        report += f"""
**Best Classification AUC**: {best_auc:.4f} ({best_exp})
- Configuration: {exp['n_ahead']}-possession-ahead, {exp['n_states']} states
- Improvement: {exp['results']['improvement']['auc']:+.4f} over baseline
"""

    # Find best R²
    best_r2 = -999
    best_regr = None
    for exp_name, exp in results['experiments'].items():
        if exp['task'] == 'regression':
            r2 = exp['results']['with_momentum']['r2']
            if r2 > best_r2:
                best_r2 = r2
                best_regr = exp_name

    if best_regr:
        exp = results['experiments'][best_regr]
        report += f"""
**Best Regression R²**: {best_r2:.4f} ({best_regr})
- Configuration: {exp['n_ahead']}-possession-ahead, {exp['n_states']} states
- Baseline R²: {exp['results']['baseline']['r2']:.4f}
- Improvement: {exp['results']['improvement']['r2_improvement']:+.4f}
"""

    report += """
### Insights

1. **Prediction Horizon**: Longer prediction windows (3-5 possessions) may be more stable
2. **Task Type**: Regression vs classification trade-offs
3. **State Count**: Optimal balance between granularity and sample size

---

*Generated automatically by run_all_experiments.py*
"""

    with open('EXPERIMENT_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\n✓ Summary report saved to EXPERIMENT_REPORT.md")


if __name__ == "__main__":
    results = run_all_experiments()
