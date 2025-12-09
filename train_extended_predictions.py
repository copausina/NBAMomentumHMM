"""
Extended Predictions: Complete Proposal Implementation

Implements all prediction targets from proposal:
1. Next possession scoring (already done)
2. Next n possessions scoring (NEW)
3. Win prediction (NEW)
4. Permutation tests for significance (NEW)
"""
import sys
sys.path.append('/Users/xavier/Downloads/NBAMomentumHMM-main')

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.utils import shuffle
from scipy.stats import f_oneway
from imblearn.over_sampling import SMOTE
from train_hmm_enhanced import create_enhanced_features, MomentumHMM

print("=" * 80)
print("EXTENDED PREDICTIONS: Complete Proposal Implementation")
print("=" * 80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('possessions_multi_season_with_players.csv')
df_enhanced = create_enhanced_features(df)
print(f"   Loaded {len(df):,} possessions from {df['gameId'].nunique()} games")

# === HMM Features ===
print("\n2. Training HMM...")
hmm_features = [
    'pointsLast3', 'pointsLast5', 'runMagnitude',
    'shotDistance',
    'foulsCommitted', 'foulsDrawn', 'foulAdvantage',
    'oRebsGrabbed', 'secondChance',
    'scoreDiffStart', 'period', 'clockStart', 'periodPressure', 'gameImportance',
    'team_shooting_streak', 'team_rolling_ppg', 'avg_player_fg_pct', 'avg_hot_hand_pct',
]

hmm = MomentumHMM(n_states=3, n_iter=100, random_state=42)
X, lengths = hmm.prepare_sequences(df_enhanced, hmm_features)
hmm.fit(X, lengths)

states = hmm.predict_states(X, lengths)
df_enhanced['state'] = states

print(f"   ✓ HMM trained with {len(hmm_features)} features")

# === Create ALL prediction targets ===
print("\n3. Creating prediction targets...")

df_eval = df_enhanced.copy()

# Target 1: Next possession scoring (existing)
df_eval['y_next_1'] = (df_eval.groupby('gameId')['pointsScored'].shift(-1) > 0).astype(float)

# Target 2: Next 3 possessions scoring (NEW)
df_eval['scored_next_3'] = df_eval.groupby('gameId')['pointsScored'].shift(-1).rolling(window=3, min_periods=1).sum()
df_eval['y_next_3'] = (df_eval['scored_next_3'] > 0).astype(float)

# Target 3: Next 5 possessions scoring (NEW)
df_eval['scored_next_5'] = df_eval.groupby('gameId')['pointsScored'].shift(-1).rolling(window=5, min_periods=1).sum()
df_eval['y_next_5'] = (df_eval['scored_next_5'] > 0).astype(float)

# Target 4: Team wins game (NEW)
def get_game_outcome(group):
    """Determine if team won based on final score differential"""
    final_score_diff = group['scoreDiffStart'].iloc[-1] + group['pointsScored'].iloc[-1] - group['pointsAllowed'].iloc[-1]
    return final_score_diff > 0

df_eval['team_won'] = df_eval.groupby('gameId').apply(
    lambda g: get_game_outcome(g)
).reindex(df_eval.index, method='ffill').astype(float)

print("   Created targets:")
print(f"   - Next 1 possession:  {df_eval['y_next_1'].notna().sum():,} samples")
print(f"   - Next 3 possessions: {df_eval['y_next_3'].notna().sum():,} samples")
print(f"   - Next 5 possessions: {df_eval['y_next_5'].notna().sum():,} samples")
print(f"   - Win prediction:     {df_eval['team_won'].notna().sum():,} samples")

# === Prepare causal features ===
print("\n4. Preparing causal features...")

df_eval['prev_state'] = df_eval.groupby('gameId')['state'].shift(1)
df_eval['prev_scoreDiff'] = df_eval.groupby('gameId')['scoreDiffStart'].shift(1)
df_eval['prev_clock'] = df_eval.groupby('gameId')['clockStart'].shift(1)

# Filter valid possessions
offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
df_eval = df_eval[df_eval["result"].isin(offense_results)]
df_eval = df_eval.dropna(subset=['prev_state', 'prev_scoreDiff', 'prev_clock'])

# Create state dummies
state_dummies = pd.get_dummies(df_eval['prev_state'], prefix='prev_momentum')
df_eval = pd.concat([df_eval, state_dummies], axis=1)

# Split
split_idx = int(len(df_eval) * 0.8)
train_df = df_eval.iloc[:split_idx]
test_df = df_eval.iloc[split_idx:]

print(f"   Train: {len(train_df):,}, Test: {len(test_df):,}")

# === Baseline and momentum features ===
baseline_features = ['prev_scoreDiff', 'prev_clock']
momentum_features = state_dummies.columns.tolist()
all_features = baseline_features + momentum_features

X_train = train_df[all_features].values
X_test = test_df[all_features].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# === PREDICTION 1: Next Possession ===
print("\n" + "=" * 80)
print("PREDICTION 1: Next Possession Scoring")
print("=" * 80)

y_train_1 = train_df['y_next_1'].dropna().values[:len(X_train_sc)]
y_test_1 = test_df['y_next_1'].dropna().values[:len(X_test_sc)]

# Ensure matching lengths
min_len = min(len(y_train_1), len(X_train_sc))
X_train_1 = X_train_sc[:min_len]
y_train_1 = y_train_1[:min_len]

min_len_test = min(len(y_test_1), len(X_test_sc))
X_test_1 = X_test_sc[:min_len_test]
y_test_1 = y_test_1[:min_len_test]

# Train with SMOTE
smote = SMOTE(random_state=42)
X_train_1_smote, y_train_1_smote = smote.fit_resample(X_train_1, y_train_1)

model_1 = LogisticRegression(max_iter=1000, random_state=42)
model_1.fit(X_train_1_smote, y_train_1_smote)
y_prob_1 = model_1.predict_proba(X_test_1)[:, 1]

auc_1 = roc_auc_score(y_test_1, y_prob_1)
ll_1 = log_loss(y_test_1, y_prob_1)

print(f"\n   AUC:      {auc_1:.4f}")
print(f"   Log-loss: {ll_1:.4f}")

# === PREDICTION 2: Next 3 Possessions ===
print("\n" + "=" * 80)
print("PREDICTION 2: Next 3 Possessions Scoring")
print("=" * 80)

y_train_3 = train_df['y_next_3'].dropna().values[:len(X_train_sc)]
y_test_3 = test_df['y_next_3'].dropna().values[:len(X_test_sc)]

min_len = min(len(y_train_3), len(X_train_sc))
X_train_3 = X_train_sc[:min_len]
y_train_3 = y_train_3[:min_len]

min_len_test = min(len(y_test_3), len(X_test_sc))
X_test_3 = X_test_sc[:min_len_test]
y_test_3 = y_test_3[:min_len_test]

X_train_3_smote, y_train_3_smote = smote.fit_resample(X_train_3, y_train_3)

model_3 = LogisticRegression(max_iter=1000, random_state=42)
model_3.fit(X_train_3_smote, y_train_3_smote)
y_prob_3 = model_3.predict_proba(X_test_3)[:, 1]

auc_3 = roc_auc_score(y_test_3, y_prob_3)
ll_3 = log_loss(y_test_3, y_prob_3)

print(f"\n   AUC:      {auc_3:.4f}")
print(f"   Log-loss: {ll_3:.4f}")

# === PREDICTION 3: Next 5 Possessions ===
print("\n" + "=" * 80)
print("PREDICTION 3: Next 5 Possessions Scoring")
print("=" * 80)

y_train_5 = train_df['y_next_5'].dropna().values[:len(X_train_sc)]
y_test_5 = test_df['y_next_5'].dropna().values[:len(X_test_sc)]

min_len = min(len(y_train_5), len(X_train_sc))
X_train_5 = X_train_sc[:min_len]
y_train_5 = y_train_5[:min_len]

min_len_test = min(len(y_test_5), len(X_test_sc))
X_test_5 = X_test_sc[:min_len_test]
y_test_5 = y_test_5[:min_len_test]

X_train_5_smote, y_train_5_smote = smote.fit_resample(X_train_5, y_train_5)

model_5 = LogisticRegression(max_iter=1000, random_state=42)
model_5.fit(X_train_5_smote, y_train_5_smote)
y_prob_5 = model_5.predict_proba(X_test_5)[:, 1]

auc_5 = roc_auc_score(y_test_5, y_prob_5)
ll_5 = log_loss(y_test_5, y_prob_5)

print(f"\n   AUC:      {auc_5:.4f}")
print(f"   Log-loss: {ll_5:.4f}")

# === PREDICTION 4: Win Prediction ===
print("\n" + "=" * 80)
print("PREDICTION 4: Win Prediction")
print("=" * 80)

# Use existing team_won column (calculated earlier in line 76-78)
y_train_win = train_df['team_won'].dropna().values
y_test_win = test_df['team_won'].dropna().values

print(f"   Train samples: {len(y_train_win)}, Test samples: {len(y_test_win)}")
print(f"   Win rate (train): {y_train_win.mean():.2%}")

if len(y_train_win) == 0 or len(y_test_win) == 0:
    print("   ⚠️  No valid data for win prediction - skipping")
else:
    # Use class_weight instead of SMOTE to handle imbalance
    model_win = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model_win.fit(X_train_sc[:len(y_train_win)], y_train_win)
    y_prob_win = model_win.predict_proba(X_test_sc[:len(y_test_win)])[:, 1]
    y_pred_win = (y_prob_win > 0.5).astype(int)

auc_win = roc_auc_score(y_test_win, y_prob_win)
acc_win = accuracy_score(y_test_win, y_pred_win)
ll_win = log_loss(y_test_win, y_prob_win)

print(f"\n   AUC:      {auc_win:.4f}")
print(f"   Accuracy: {acc_win:.4f} ({acc_win*100:.1f}%)")
print(f"   Log-loss: {ll_win:.4f}")

# === PERMUTATION TESTS ===
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE: Permutation Tests")
print("=" * 80)

print("\n   Running permutation tests (1000 iterations)...")

def permutation_test(y_true, y_pred_proba, n_permutations=1000):
    """Permutation test for AUC significance"""
    observed_auc = roc_auc_score(y_true, y_pred_proba)

    null_aucs = []
    for i in range(n_permutations):
        y_permuted = shuffle(y_true, random_state=i)
        null_auc = roc_auc_score(y_permuted, y_pred_proba)
        null_aucs.append(null_auc)

    p_value = (np.array(null_aucs) >= observed_auc).sum() / n_permutations
    return observed_auc, p_value, null_aucs

# Test 1: Next possession
print("\n   Test 1: Next Possession")
obs_auc_1, p_val_1, null_dist_1 = permutation_test(y_test_1, y_prob_1)
print(f"      Observed AUC: {obs_auc_1:.4f}")
print(f"      p-value:      {p_val_1:.4f}")
print(f"      Significant:  {'✓' if p_val_1 < 0.05 else '✗'} (p < 0.05)")

# Test 2: Next 3 possessions
print("\n   Test 2: Next 3 Possessions")
obs_auc_3, p_val_3, null_dist_3 = permutation_test(y_test_3, y_prob_3)
print(f"      Observed AUC: {obs_auc_3:.4f}")
print(f"      p-value:      {p_val_3:.4f}")
print(f"      Significant:  {'✓' if p_val_3 < 0.05 else '✗'} (p < 0.05)")

# Test 3: Next 5 possessions
print("\n   Test 3: Next 5 Possessions")
obs_auc_5, p_val_5, null_dist_5 = permutation_test(y_test_5, y_prob_5)
print(f"      Observed AUC: {obs_auc_5:.4f}")
print(f"      p-value:      {p_val_5:.4f}")
print(f"      Significant:  {'✓' if p_val_5 < 0.05 else '✗'} (p < 0.05)")

# Test 4: Win prediction
print("\n   Test 4: Win Prediction")
obs_auc_win, p_val_win, null_dist_win = permutation_test(y_test_win, y_prob_win)
print(f"      Observed AUC: {obs_auc_win:.4f}")
print(f"      p-value:      {p_val_win:.4f}")
print(f"      Significant:  {'✓' if p_val_win < 0.05 else '✗'} (p < 0.05)")

# === SUMMARY ===
print("\n" + "=" * 80)
print("COMPLETE RESULTS SUMMARY")
print("=" * 80)

results = {
    'Next 1 Possession': {'AUC': auc_1, 'Log-loss': ll_1, 'p-value': p_val_1},
    'Next 3 Possessions': {'AUC': auc_3, 'Log-loss': ll_3, 'p-value': p_val_3},
    'Next 5 Possessions': {'AUC': auc_5, 'Log-loss': ll_5, 'p-value': p_val_5},
    'Win Prediction': {'AUC': auc_win, 'Accuracy': acc_win, 'p-value': p_val_win},
}

print("\n   Prediction Performance:")
for name, metrics in results.items():
    print(f"\n   {name}:")
    for metric, value in metrics.items():
        print(f"      {metric}: {value:.4f}")

# State interpretability
state_groups = []
for state_num in range(3):
    state_data = df_enhanced[df_enhanced['state'] == state_num]
    state_groups.append(state_data['pointsScored'].values)

f_stat, p_value_anova = f_oneway(*state_groups)
print(f"\n   State Interpretability:")
print(f"      ANOVA F-stat: {f_stat:.2f}")
print(f"      ANOVA p-value: {p_value_anova:.6f}")
print(f"      Significant: {'✓' if p_value_anova < 0.05 else '✗'}")

# === Save Extended Model ===
print("\n5. Saving extended model...")

with open('extended_momentum_model.pkl', 'wb') as f:
    pickle.dump({
        'hmm': hmm,
        'features': hmm_features,
        'n_states': 3,

        # Models
        'model_next_1': model_1,
        'model_next_3': model_3,
        'model_next_5': model_5,
        'model_win': model_win,

        # Scaler
        'scaler': scaler,

        # Results
        'results': {
            'next_1': {'auc': auc_1, 'log_loss': ll_1, 'p_value': p_val_1},
            'next_3': {'auc': auc_3, 'log_loss': ll_3, 'p_value': p_val_3},
            'next_5': {'auc': auc_5, 'log_loss': ll_5, 'p_value': p_val_5},
            'win': {'auc': auc_win, 'accuracy': acc_win, 'p_value': p_val_win},
        },

        # State analysis
        'anova_pvalue': p_value_anova,
        'transition_matrix': hmm.model.transmat_,

        'config': 'Extended predictions with permutation tests',
    }, f)

print(f"   ✓ Saved to extended_momentum_model.pkl")

print("\n" + "=" * 80)
print("✓ EXTENDED PREDICTIONS COMPLETE")
print("=" * 80)

print("\n💡 Key Insights:")
print(f"   - Next 1 possession:  AUC {auc_1:.4f} (p={p_val_1:.4f})")
print(f"   - Next 3 possessions: AUC {auc_3:.4f} (p={p_val_3:.4f})")
print(f"   - Next 5 possessions: AUC {auc_5:.4f} (p={p_val_5:.4f})")
print(f"   - Win prediction:     AUC {auc_win:.4f}, Acc {acc_win*100:.1f}% (p={p_val_win:.4f})")
print(f"\n   All proposal prediction targets now implemented!")
