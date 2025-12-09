# NBA Momentum Analysis with Hidden Markov Models

3-state HMM modeling basketball momentum across 164 NBA games (2022-24 seasons).

## Results

| Metric | Value |
|--------|-------|
| **AUC** | 0.5186 (baseline: 0.5101) |
| **Improvement** | +0.0086 (+1.7%) |
| **Dataset** | 33,157 possessions, 164 games |
| **State Interpretability** | ANOVA p < 0.000001 ✓ |

### Momentum States

| State | % | Scoring Rate | Avg Points |
|-------|---|--------------|------------|
| COLD | 9.1% | 1.3% | 0.03 |
| NEUTRAL | 77.0% | 21.4% | 0.47 |
| HOT | 13.9% | 66.4% | 1.45 |

## Setup

```bash
conda create -n nba-hmm python=3.10 -y
conda activate nba-hmm
pip install nba_api pandas numpy scikit-learn hmmlearn imbalanced-learn scipy wandb flask plotly
```

## Quick Start

### 1. Train the Model

```bash
python download_multi_seasons.py          # Download data
python feature_engineer_v2.py             # Extract features
python add_player_features_simple.py      # Add player stats
python train_causal_momentum.py           # Train HMM
```

Output: `final_momentum_model.pkl`

### 2. Launch Web UI

**Optional: Fetch team names first (enhances game descriptions)**
```bash
python fetch_game_teams.py
```

**Start the web server:**
```bash
cd web
python app.py
```

The server will start on `http://localhost:5000`. You should see:
```
Training win probability model...
Win probability model accuracy: 74.5%
Running on http://127.0.0.1:5000
```

**Access the visualization:**
Open your browser and navigate to **http://localhost:5000**

Features available:
- **Game-by-game momentum timelines** - Visualize HOT/COLD/NEUTRAL states across possessions
- **ML-powered win probability** - Trained logistic regression (74.5% accuracy) with deterministic endgame
- **Possession-level analysis** - Score tracking, momentum shifts, and result breakdowns

## Advanced Features

### W&B Integration

Track experiments with Weights & Biases:

```bash
python train_causal_momentum_wandb.py
```

Logs metrics, transition matrices, and model artifacts to your wandb dashboard.

### Extended Predictions

Train models for additional prediction targets:

```bash
python train_extended_predictions.py
```

Implements:
- Next 1 possession scoring (AUC ~0.52)
- Next 3 possessions scoring
- Next 5 possessions scoring
- Win prediction
- Permutation tests for significance

Output: `extended_momentum_model.pkl`

### Team Context (Optional)

Add schedule-based features from basketball-reference.com:

```bash
python fetch_team_schedules.py
```

Enriches data with: team records, win streaks, back-to-back games, days of rest.

## Project Structure

```
download_multi_seasons.py              # NBA API data collection
feature_engineer_v2.py                 # Possession-level features
add_player_features_simple.py          # Player stat aggregation
train_hmm.py                           # HMM class implementation
train_hmm_enhanced.py                  # Feature engineering helpers
train_causal_momentum.py               # Main training script
train_causal_momentum_wandb.py         # W&B integration
train_extended_predictions.py          # Multi-target models
fetch_team_schedules.py                # Schedule scraper
fetch_game_teams.py                    # Team name fetcher
final_momentum_model.pkl               # Trained HMM model
web/                                   # Interactive visualization
  ├── app.py                           # Flask backend + ML win predictor
  ├── templates/index.html             # Frontend UI
  ├── static/js/main.js                # Client-side logic
  └── static/css/style.css             # Styling
experiments/                           # Optimization experiments
```

## Key Findings

✅ **HMM successfully identifies three interpretable momentum states** with statistically significant differences in scoring behavior (ANOVA p < 0.000001).

⚠️ **Momentum provides minimal predictive power** (AUC 0.5186 vs baseline 0.5101), showing single-possession outcomes in basketball are highly stochastic.

📚 **Results align with "hot hand fallacy" literature** - momentum effects exist psychologically and statistically, but are too weak for reliable game prediction.

🎯 **Web UI demonstrates practical application** - ML-based win probability model achieves 74.5% accuracy by incorporating momentum states, score differential, and time context.

## Technical Notes

- **Causal inference**: Uses state at t-1 to predict outcome at t (no data leakage)
- **Features**: 18 historical indicators including scoring, fouls, rebounds, context, and player stats
- **Model**: 3-state Gaussian HMM with diagonal covariance
- **Win probability**: Logistic regression trained on 33,157 possessions
  - Features: momentum state (one-hot), score diff, time remaining, period
  - Blends ML prediction with deterministic formula in final 5 possessions
  - Ensures 100%/0% confidence at game end
- **Warning**: Feature `team_shooting_streak` has perfect correlation with target (AUC=1.0) and must be excluded to avoid data leakage

## Citation

This project implements causal momentum analysis using Hidden Markov Models to test the "hot hand" hypothesis in NBA basketball, with an interactive web interface for exploring momentum dynamics across 164 games.
