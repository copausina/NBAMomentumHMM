# NBA Momentum Analysis with Hidden Markov Models

3-state HMM modeling basketball momentum across 164 NBA games (2022-24 seasons).

## Results

| Metric | Value |
|--------|-------|
| **AUC** | 0.5186 (baseline: 0.5101) |
| **Improvement** | +0.0086 (+1.7%) |
| **Dataset** | 33,157 possessions |
| **State Interpretability** | ANOVA p < 0.000001 ✓ |

### Momentum States

| State | % | Scoring Rate |
|-------|---|--------------|
| COLD | 9.1% | 1.3% |
| NEUTRAL | 77.0% | 21.4% |
| HOT | 13.9% | 66.4% |

## Setup

```bash
conda create -n nba-hmm python=3.10 -y
conda activate nba-hmm
pip install nba_api pandas numpy scikit-learn hmmlearn imbalanced-learn scipy
```

## Pipeline

```bash
python download_multi_seasons.py          # Download data
python feature_engineer_v2.py             # Extract features
python add_player_features_simple.py      # Add player stats
python train_causal_momentum.py           # Train model
```

Output: `final_momentum_model.pkl`

## Files

```
download_multi_seasons.py              # Data collection
feature_engineer_v2.py                 # Feature extraction
add_player_features_simple.py          # Player features
train_hmm.py                           # HMM class (dependency)
train_hmm_enhanced.py                  # Enhanced features (dependency)
train_causal_momentum.py               # Main training script
final_momentum_model.pkl               # Trained model
```

## Key Findings

✅ HMM identifies three interpretable momentum states with significantly different scoring characteristics.

⚠️ Momentum provides minimal predictive improvement (AUC 0.52 vs 0.51), indicating basketball single possession outcomes are highly stochastic.

📚 Results consistent with "hot hand fallacy" literature - momentum effects exist but are too weak for reliable prediction.

## Technical Notes

- **Causal inference**: Uses state at t-1 to predict outcome at t
- **Features**: 18 historical indicators (scoring, fouls, rebounds, context, player stats)
- **No data leakage**: Excludes current possession outcome from HMM training
- **Warning**: `team_shooting_streak` has perfect correlation with target (AUC=1.0) and must be excluded
