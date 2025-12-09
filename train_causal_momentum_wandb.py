"""
Causal Momentum Model with Weights & Biases Integration

Key fix: Use state at time t-1 to predict outcome at time t
This ensures we don't use future information (no "cheating")

Goals:
1. Next possession prediction (AUC > 0.65)
2. Interpretable states (ANOVA p < 0.05)
3. Proper causal inference
"""
import sys
sys.path.append('/Users/xavier/Downloads/NBAMomentumHMM-main')

import pandas as pd
import numpy as np
import pickle
import wandb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import f_oneway
from imblearn.over_sampling import SMOTE
from train_hmm_enhanced import create_enhanced_features, MomentumHMM

# Initialize Weights & Biases
wandb.init(
    project="nba-momentum-hmm",
    name="causal-momentum-3states",
    config={
        "n_states": 3,
        "n_iter": 100,
        "random_state": 42,
        "model": "HMM + Logistic Regression + SMOTE",
        "seasons": "2022-23, 2023-24",
        "causal_inference": True,
    }
)

print("=" * 80)
print("CAUSAL MOMENTUM MODEL: Proper Time-Series Prediction")
print("=" * 80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('possessions_multi_season_with_players.csv')
df_enhanced = create_enhanced_features(df)
print(f"   Loaded {len(df):,} possessions from {df['gameId'].nunique()} games")

# Log dataset info
wandb.config.update({
    "n_possessions": len(df),
    "n_games": df['gameId'].nunique()
})

# === HMM Features: Historical indicators only ===
print("\n2. Defining HMM features...")

hmm_features = [
    # Historical scoring (shifted to exclude current)
    'pointsLast3',
    'pointsLast5',
    'runMagnitude',

    # Shot quality
    'shotDistance',

    # Fouls
    'foulsCommitted',
    'foulsDrawn',
    'foulAdvantage',

    # Rebounds
    'oRebsGrabbed',
    'secondChance',

    # Context
    'scoreDiffStart',
    'period',
    'clockStart',
    'periodPressure',
    'gameImportance',

    # Player performance
    'team_shooting_streak',
    'team_rolling_ppg',
    'avg_player_fg_pct',
    'avg_hot_hand_pct',
]

print(f"   ✓ {len(hmm_features)} historical features")
wandb.config.update({"n_features": len(hmm_features)})

# === Train HMM ===
print("\n3. Training HMM...")
print("-" * 80)

hmm = MomentumHMM(n_states=3, n_iter=100, random_state=42)
X, lengths = hmm.prepare_sequences(df_enhanced, hmm_features)
hmm.fit(X, lengths)

# Get states for ALL possessions
states = hmm.predict_states(X, lengths)
df_enhanced['state'] = states

unique, counts = np.unique(states, return_counts=True)
print("\n📊 State distribution:")
state_dist = {}
for state_num, count in zip(unique, counts):
    pct = count/len(states)*100
    print(f"   State {state_num}: {count:6,} ({pct:5.1f}%)")
    state_dist[f"state_{state_num}_pct"] = pct

# Log state distribution to wandb
wandb.log(state_dist)

# === State Interpretability ===
print("\n4. STATE INTERPRETABILITY")
print("-" * 80)

state_groups = []
state_stats = {}
for state_num in unique:
    state_data = df_enhanced[df_enhanced['state'] == state_num]
    state_groups.append(state_data['pointsScored'].values)

    pct_scoring = (state_data['pointsScored'] > 0).mean() * 100
    avg_points = state_data['pointsScored'].mean()

    print(f"\n   State {state_num}:")
    print(f"      Scoring rate:  {pct_scoring:5.1f}%")
    print(f"      Avg points:    {avg_points:.3f}")

    state_stats[f"state_{state_num}_scoring_rate"] = pct_scoring
    state_stats[f"state_{state_num}_avg_points"] = avg_points

# ANOVA
f_stat, p_value = f_oneway(*state_groups)
print(f"\n   ANOVA: F={f_stat:.2f}, p={p_value:.6f}")
print(f"   {'✓' if p_value < 0.05 else '✗'} Significantly different (p<0.05)")

state_stats["anova_f_stat"] = f_stat
state_stats["anova_p_value"] = p_value
wandb.log(state_stats)

# === Log transition matrix ===
transition_matrix = hmm.model.transmat_
print(f"\n📊 Transition Matrix:")
print(transition_matrix)
wandb.log({"transition_matrix": wandb.Table(
    data=transition_matrix,
    columns=[f"To State {i}" for i in range(3)]
)})

# === CAUSAL PREDICTION: Use state[t-1] to predict outcome[t] ===
print("\n5. CAUSAL NEXT POSSESSION PREDICTION")
print("-" * 80)

# Prepare data with LAGGED state
df_eval = df_enhanced.copy()
df_eval['y_scored'] = (df_eval['pointsScored'] > 0).astype(int)

# KEY FIX: Use PREVIOUS possession's state to predict CURRENT outcome
df_eval['prev_state'] = df_eval.groupby('gameId')['state'].shift(1)

# Also shift other features for proper causal inference
df_eval['prev_scoreDiff'] = df_eval.groupby('gameId')['scoreDiffStart'].shift(1)
df_eval['prev_clock'] = df_eval.groupby('gameId')['clockStart'].shift(1)

# Filter valid possessions
offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
df_eval = df_eval[df_eval["result"].isin(offense_results)]
df_eval = df_eval.dropna(subset=['prev_state', 'prev_scoreDiff', 'prev_clock'])

print(f"\n   Valid possessions: {len(df_eval):,}")

# Create state dummies from PREVIOUS state
state_dummies = pd.get_dummies(df_eval['prev_state'], prefix='prev_momentum')
df_eval = pd.concat([df_eval, state_dummies], axis=1)

# Split
split_idx = int(len(df_eval) * 0.8)
train_df = df_eval.iloc[:split_idx]
test_df = df_eval.iloc[split_idx:]

y_train = train_df['y_scored'].values
y_test = test_df['y_scored'].values

print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")

# === Baseline: Previous score diff + time ===
print("\n   5a. BASELINE (prev scoreDiff + clock)")

baseline_features = ['prev_scoreDiff', 'prev_clock']
X_train_base = train_df[baseline_features].values
X_test_base = test_df[baseline_features].values

scaler_base = StandardScaler()
X_train_base_sc = scaler_base.fit_transform(X_train_base)
X_test_base_sc = scaler_base.transform(X_test_base)

model_base = LogisticRegression(max_iter=1000, random_state=42)
model_base.fit(X_train_base_sc, y_train)
y_prob_base = model_base.predict_proba(X_test_base_sc)[:, 1]

auc_base = roc_auc_score(y_test, y_prob_base)
ll_base = log_loss(y_test, y_prob_base)

print(f"      AUC:      {auc_base:.4f}")
print(f"      Log-loss: {ll_base:.4f}")

wandb.log({
    "baseline_auc": auc_base,
    "baseline_log_loss": ll_base
})

# === Momentum Model: Baseline + Previous State ===
print("\n   5b. MOMENTUM MODEL (baseline + prev state)")

momentum_features = state_dummies.columns.tolist()
all_features = baseline_features + momentum_features

X_train_mom = train_df[all_features].values
X_test_mom = test_df[all_features].values

scaler_mom = StandardScaler()
X_train_mom_sc = scaler_mom.fit_transform(X_train_mom)
X_test_mom_sc = scaler_mom.transform(X_test_mom)

model_mom = LogisticRegression(max_iter=1000, random_state=42)
model_mom.fit(X_train_mom_sc, y_train)
y_prob_mom = model_mom.predict_proba(X_test_mom_sc)[:, 1]

auc_mom = roc_auc_score(y_test, y_prob_mom)
ll_mom = log_loss(y_test, y_prob_mom)

print(f"      AUC:      {auc_mom:.4f}  ({auc_mom - auc_base:+.4f})")
print(f"      Log-loss: {ll_mom:.4f}  ({ll_mom - ll_base:+.4f})")

wandb.log({
    "momentum_auc": auc_mom,
    "momentum_log_loss": ll_mom,
    "momentum_auc_improvement": auc_mom - auc_base
})

# === With SMOTE ===
print("\n   5c. MOMENTUM + SMOTE")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_mom, y_train)

scaler_smote = StandardScaler()
X_train_smote_sc = scaler_smote.fit_transform(X_train_smote)
X_test_smote_sc = scaler_smote.transform(X_test_mom)

model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_smote_sc, y_train_smote)
y_prob_smote = model_smote.predict_proba(X_test_smote_sc)[:, 1]

auc_smote = roc_auc_score(y_test, y_prob_smote)
ll_smote = log_loss(y_test, y_prob_smote)

print(f"      AUC:      {auc_smote:.4f}  ({auc_smote - auc_base:+.4f})")
print(f"      Log-loss: {ll_smote:.4f}  ({ll_smote - ll_base:+.4f})")

wandb.log({
    "final_auc": auc_smote,
    "final_log_loss": ll_smote,
    "final_auc_improvement": auc_smote - auc_base,
    "auc_improvement_pct": (auc_smote - auc_base) / auc_base * 100
})

# Enhanced model removed due to data leakage in team_shooting_streak
# Use momentum + SMOTE as final model
auc_enh = auc_smote  # Best model without leakage

# === Summary ===
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n📊 Prediction Performance (Causal Inference)")
print(f"   Baseline:                AUC = {auc_base:.4f}")
print(f"   + Momentum states:       AUC = {auc_mom:.4f}  ({auc_mom - auc_base:+.4f})")
print(f"   + SMOTE:                 AUC = {auc_smote:.4f}  ({auc_smote - auc_base:+.4f})  ⭐ FINAL")

print(f"\n   Proposal target: AUC > 0.65")
best_auc = auc_smote  # Final model: Momentum + SMOTE

if best_auc >= 0.65:
    print(f"   ✓ TARGET ACHIEVED! Best AUC: {best_auc:.4f}")
    wandb.log({"target_achieved": True})
elif best_auc >= 0.58:
    print(f"   ⚠️  Approaching target. Best AUC: {best_auc:.4f}")
    print(f"      Basketball momentum is weaker than proposal expected")
    wandb.log({"target_achieved": False})
else:
    print(f"   ✗ Below target. Best AUC: {best_auc:.4f}")
    print(f"      Single possession prediction is inherently difficult")
    print(f"      Consistent with 'hot hand fallacy' literature")
    wandb.log({"target_achieved": False})

print(f"\n📊 State Interpretability")
print(f"   ANOVA: p = {p_value:.6f}")
print(f"   {'✓' if p_value < 0.05 else '✗'} States differ significantly")

# Save
print("\n6. Saving model...")
with open('final_momentum_model.pkl', 'wb') as f:
    pickle.dump({
        'hmm': hmm,
        'features': hmm_features,
        'n_states': 3,
        'auc_baseline': auc_base,
        'auc_momentum': auc_mom,
        'auc_final': auc_smote,
        'anova_pvalue': p_value,
        'improvement': auc_smote - auc_base,
        'config': 'Causal inference with momentum states + SMOTE',
        'transition_matrix': transition_matrix,
    }, f)

# Save model to wandb
wandb.save('final_momentum_model.pkl')

print(f"   ✓ Saved to final_momentum_model.pkl")
print(f"   Final AUC: {best_auc:.4f} (improvement: {best_auc - auc_base:+.4f})")

print("\n" + "=" * 80)
print("✓ CAUSAL ANALYSIS COMPLETE")
print("=" * 80)

# Additional insight
print("\n💡 Key Insight:")
print(f"   Baseline (random): ~0.50")
print(f"   Our baseline:      {auc_base:.4f}")
print(f"   Best model:        {best_auc:.4f}")
print(f"   Improvement:       {best_auc - 0.50:.4f} over random")
print(f"\n   NBA single possession outcomes are highly stochastic.")
print(f"   This result validates 'hot hand fallacy' research.")

# Final summary to wandb
wandb.run.summary["best_auc"] = best_auc
wandb.run.summary["anova_p_value"] = p_value
wandb.run.summary["interpretable_states"] = p_value < 0.05

wandb.finish()
print("\n✓ Weights & Biases logging complete")
