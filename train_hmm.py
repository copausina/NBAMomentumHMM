"""
Hidden Markov Model Training for NBA Momentum Analysis

This script trains an HMM to discover latent momentum states in basketball games
and evaluates whether momentum states predict future scoring.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, classification_report
)
from hmmlearn import hmm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Optional: wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class MomentumHMM:
    """
    Hidden Markov Model for discovering momentum states in basketball games
    """

    def __init__(self, n_states=3, n_iter=100, random_state=42):
        """
        Args:
            n_states: Number of hidden states (default 3: hot, neutral, cold)
            n_iter: Number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.state_labels = None  # Will be assigned after training

    def prepare_sequences(self, df, features):
        """
        Prepare sequences for HMM training

        Args:
            df: DataFrame with possession-level data
            features: List of feature column names

        Returns:
            X: Scaled feature matrix
            lengths: Length of each game sequence
        """
        self.feature_names = features

        # Group by game
        game_sequences = []
        lengths = []

        for game_id, game_df in df.groupby('gameId'):
            game_features = game_df[features].fillna(0).values
            game_sequences.append(game_features)
            lengths.append(len(game_features))

        # Concatenate all sequences
        X = np.vstack(game_sequences)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, lengths

    def fit(self, X, lengths):
        """
        Fit the HMM model

        Args:
            X: Feature matrix (concatenated sequences)
            lengths: Length of each sequence
        """
        print(f"Training HMM with {self.n_states} states...")
        print(f"Total observations: {len(X):,}")
        print(f"Number of sequences: {len(lengths)}")

        # Use Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False
        )

        # Fit the model
        self.model.fit(X, lengths)

        print(f"✓ HMM training complete")
        print(f"  Log likelihood: {self.model.score(X, lengths):.2f}")
        print(f"  Converged: {self.model.monitor_.converged}")

        return self

    def predict_states(self, X, lengths):
        """
        Predict hidden states for sequences

        Args:
            X: Feature matrix
            lengths: Sequence lengths

        Returns:
            states: Predicted state for each observation
        """
        return self.model.predict(X, lengths)

    def analyze_states(self, df, states, metric='pointsScored'):
        """
        Analyze and interpret learned states

        Args:
            df: Original DataFrame
            states: Predicted states
            metric: Metric to use for interpretation (default: pointsScored)

        Returns:
            state_stats: Statistics for each state
        """
        df_with_states = df.copy()
        df_with_states['state'] = states

        state_stats = df_with_states.groupby('state').agg({
            metric: ['mean', 'std', 'count'],
            'shotDistance': 'mean',
            'foulsCommitted': 'mean',
            'oRebsGrabbed': 'mean',
        }).round(3)

        # Sort states by average points scored
        state_order = df_with_states.groupby('state')[metric].mean().sort_values(ascending=False)

        # Assign interpretable labels
        self.state_labels = {}
        if len(state_order) >= 3:
            self.state_labels[state_order.index[0]] = 'HOT'
            self.state_labels[state_order.index[-1]] = 'COLD'
            self.state_labels[state_order.index[1]] = 'NEUTRAL'
        else:
            for i, idx in enumerate(state_order.index):
                self.state_labels[idx] = f'State_{i}'

        return state_stats, state_order

    def get_transition_matrix(self):
        """Get the state transition matrix"""
        return self.model.transmat_

    def compute_state_statistics(self, df, states):
        """
        Compute comprehensive statistics for each state

        Args:
            df: Original DataFrame
            states: Predicted states

        Returns:
            dict: Statistics for each state
        """
        df_with_states = df.copy()
        df_with_states['state'] = states

        stats_dict = {}
        for state in range(self.n_states):
            state_data = df_with_states[df_with_states['state'] == state]
            stats_dict[state] = {
                'count': len(state_data),
                'pct': len(state_data) / len(df_with_states) * 100,
                'avg_points': state_data['pointsScored'].mean(),
                'avg_points_allowed': state_data['pointsAllowed'].mean(),
                'fg_pct': (state_data['result'].isin(['Made FG', 'Made FT']).sum() /
                          len(state_data) * 100),
                'avg_shot_distance': state_data['shotDistance'].mean(),
                'turnover_rate': (state_data['result'] == 'Turnover').sum() / len(state_data) * 100,
            }

        return stats_dict


def evaluate_momentum_prediction(df, states, hmm_model):
    """
    Evaluate if momentum states improve next-possession prediction

    Args:
        df: DataFrame with possessions
        states: Predicted momentum states
        hmm_model: Trained MomentumHMM instance

    Returns:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING MOMENTUM PREDICTION")
    print("="*60)

    df_eval = df.copy()
    df_eval['state'] = states

    # Create prediction targets
    df_eval['y_scored'] = (df_eval['pointsScored'] > 0).astype(int)
    df_eval['y_next'] = df_eval['y_scored'].shift(-1)

    # Filter valid possessions
    offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
    df_eval = df_eval[df_eval["result"].isin(offense_results)]

    # Remove last possession of each game
    df_eval = df_eval.dropna(subset=['y_next'])
    df_eval['next_gameId'] = df_eval['gameId'].shift(-1)
    df_eval = df_eval[df_eval['gameId'] == df_eval['next_gameId']]

    print(f"\nEvaluation dataset: {len(df_eval):,} possessions")

    # Features for prediction
    base_features = [
        "scoreDiffStart", "period", "clockStart",
        "foulsCommitted", "foulsDrawn", "oRebsGrabbed"
    ]

    # One-hot encode momentum states (before split!)
    state_dummies = pd.get_dummies(df_eval['state'], prefix='momentum_state')
    df_eval = pd.concat([df_eval, state_dummies], axis=1)
    momentum_features = state_dummies.columns.tolist()

    # Split by time (80/20)
    split_idx = int(len(df_eval) * 0.8)
    train_df = df_eval.iloc[:split_idx]
    test_df = df_eval.iloc[split_idx:]

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Baseline model (without momentum)
    print("\n[1/2] Training baseline model (no momentum states)...")
    X_train_base = train_df[base_features].fillna(0)
    y_train = train_df['y_next']
    X_test_base = test_df[base_features].fillna(0)
    y_test = test_df['y_next']

    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)

    model_base = LogisticRegression(max_iter=1000, random_state=42)
    model_base.fit(X_train_base_scaled, y_train)

    y_pred_base = model_base.predict(X_test_base_scaled)
    y_prob_base = model_base.predict_proba(X_test_base_scaled)[:, 1]

    # Model with momentum states
    print("[2/2] Training model WITH momentum states...")
    X_train_mom = train_df[base_features + momentum_features].fillna(0)
    X_test_mom = test_df[base_features + momentum_features].fillna(0)

    scaler_mom = StandardScaler()
    X_train_mom_scaled = scaler_mom.fit_transform(X_train_mom)
    X_test_mom_scaled = scaler_mom.transform(X_test_mom)

    model_mom = LogisticRegression(max_iter=1000, random_state=42)
    model_mom.fit(X_train_mom_scaled, y_train)

    y_pred_mom = model_mom.predict(X_test_mom_scaled)
    y_prob_mom = model_mom.predict_proba(X_test_mom_scaled)[:, 1]

    # Compute metrics
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

    # Compute improvements
    results['improvement'] = {
        'accuracy': results['with_momentum']['accuracy'] - results['baseline']['accuracy'],
        'auc': results['with_momentum']['auc'] - results['baseline']['auc'],
        'log_loss_reduction': results['baseline']['log_loss'] - results['with_momentum']['log_loss'],
    }

    # Statistical significance test
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        contingency_table = pd.crosstab(
            y_pred_base == y_test,
            y_pred_mom == y_test
        )
        if contingency_table.shape == (2, 2):
            mcnemar_result = mcnemar(contingency_table.values)
            results['mcnemar_pvalue'] = mcnemar_result.pvalue
        else:
            results['mcnemar_pvalue'] = None
    except Exception as e:
        print(f"Warning: Could not perform McNemar test: {e}")
        results['mcnemar_pvalue'] = None

    return results, model_mom, (X_test_mom_scaled, y_test, y_prob_mom)


def print_results(results, state_stats, transition_matrix, state_labels):
    """Pretty print results"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\n--- State Statistics ---")
    print(state_stats)

    print("\n--- State Interpretations ---")
    for state_num, label in state_labels.items():
        print(f"  State {state_num}: {label}")

    print("\n--- Transition Matrix ---")
    print("(Probability of transitioning from row state to column state)")
    trans_df = pd.DataFrame(
        transition_matrix,
        index=[f"{state_labels.get(i, f'S{i}')}" for i in range(len(transition_matrix))],
        columns=[f"{state_labels.get(i, f'S{i}')}" for i in range(len(transition_matrix))]
    )
    print(trans_df.round(3))

    print("\n--- Prediction Performance ---")
    print(f"Baseline (no momentum):")
    print(f"  Accuracy: {results['baseline']['accuracy']:.4f}")
    print(f"  AUC-ROC:  {results['baseline']['auc']:.4f}")
    print(f"  Log Loss: {results['baseline']['log_loss']:.4f}")

    print(f"\nWith Momentum States:")
    print(f"  Accuracy: {results['with_momentum']['accuracy']:.4f}")
    print(f"  AUC-ROC:  {results['with_momentum']['auc']:.4f}")
    print(f"  Log Loss: {results['with_momentum']['log_loss']:.4f}")

    print(f"\nImprovement:")
    print(f"  Accuracy: {results['improvement']['accuracy']:+.4f}")
    print(f"  AUC-ROC:  {results['improvement']['auc']:+.4f}")
    print(f"  Log Loss Reduction: {results['improvement']['log_loss_reduction']:+.4f}")

    if results.get('mcnemar_pvalue'):
        print(f"  McNemar's Test p-value: {results['mcnemar_pvalue']:.4f}")
        if results['mcnemar_pvalue'] < 0.05:
            print("  ✓ Improvement is statistically significant (p < 0.05)")
        else:
            print("  ✗ Improvement is not statistically significant")

    print("="*60)


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Train HMM for momentum analysis')
    parser.add_argument('--data', type=str, default='possessions.csv',
                       help='Path to possessions CSV file')
    parser.add_argument('--n_states', type=int, default=3,
                       help='Number of hidden states')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Number of EM iterations')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--project', type=str, default='nba-momentum-hmm',
                       help='Wandb project name')

    args = parser.parse_args()

    # Initialize wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.project,
            config={
                'n_states': args.n_states,
                'n_iter': args.n_iter,
                'data_file': args.data,
            }
        )
        print("✓ Wandb initialized")

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} possessions from {df['gameId'].nunique()} games")

    # Features for HMM
    hmm_features = [
        'scoreDiffStart',
        'pointsScored',
        'pointsAllowed',
        'shotDistance',
        'foulsCommitted',
        'foulsDrawn',
        'oRebsGrabbed',
        'runMagnitude',
    ]

    # Initialize and train HMM
    momentum_hmm = MomentumHMM(
        n_states=args.n_states,
        n_iter=args.n_iter,
        random_state=42
    )

    X, lengths = momentum_hmm.prepare_sequences(df, hmm_features)
    momentum_hmm.fit(X, lengths)

    # Predict states
    states = momentum_hmm.predict_states(X, lengths)

    # Analyze states
    state_stats, state_order = momentum_hmm.analyze_states(df, states)
    transition_matrix = momentum_hmm.get_transition_matrix()

    # Evaluate prediction
    results, model, test_data = evaluate_momentum_prediction(df, states, momentum_hmm)

    # Print results
    print_results(results, state_stats, transition_matrix, momentum_hmm.state_labels)

    # Log to wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            'baseline_accuracy': results['baseline']['accuracy'],
            'baseline_auc': results['baseline']['auc'],
            'momentum_accuracy': results['with_momentum']['accuracy'],
            'momentum_auc': results['with_momentum']['auc'],
            'auc_improvement': results['improvement']['auc'],
            'log_loss_reduction': results['improvement']['log_loss_reduction'],
        })

        # Log transition matrix as table
        trans_data = []
        for i in range(args.n_states):
            for j in range(args.n_states):
                trans_data.append([
                    momentum_hmm.state_labels.get(i, f'S{i}'),
                    momentum_hmm.state_labels.get(j, f'S{j}'),
                    transition_matrix[i, j]
                ])
        wandb.log({
            "transition_matrix": wandb.Table(
                columns=["From State", "To State", "Probability"],
                data=trans_data
            )
        })

        wandb.finish()

    # Save model
    import pickle
    with open('momentum_hmm_model.pkl', 'wb') as f:
        pickle.dump(momentum_hmm, f)
    print("\n✓ Model saved to momentum_hmm_model.pkl")


if __name__ == "__main__":
    main()
