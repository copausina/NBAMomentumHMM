import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from hmmlearn import hmm
import matplotlib.pyplot as plt


df = pd.read_csv('possessions.csv')
df_naive = df.copy()
df_naive["hadShot"] = df_naive["shotDistance"].notna().astype(int)
df_naive["shotDistance"] = df_naive["shotDistance"].fillna(-1)
df_naive["isHome"] = (df_naive["isHome"] == "h").astype(int)

# Binary target: did team score more points than they allowed?
df_naive["y"] = (df_naive["pointsScored"] > df_naive["pointsAllowed"]).astype(int)

# Also will filter out all possessions that ended on end of period
offense_results = ["Made FG", "Made FT", "Opp Defensive Rebound", "Turnover"]
df_naive_off = df_naive[df_naive["result"].isin(offense_results)]

df_naive_off["y_next"] = df_naive_off["y"].shift(-1)
df_naive_off = df_naive_off.dropna(subset=["y_next"])
df_naive_off["next_gameId"] = df_naive_off["gameId"].shift(-1)
df_naive_off = df_naive_off[df_naive_off["gameId"] == df_naive_off["next_gameId"]]
df_naive_off = df_naive_off.drop(columns=["next_gameId"])   
FEATURES = [
    "isHome",
    "period",
    "clockStart",
    "duration",
    "scoreDiffStart",
    "foulsCommitted",
    "foulsDrawn",
    "shotDistance",
    "oRebsGrabbed",
    # "oRebsAllowed",
    "timeouts",
    "timeoutsOpp",
    # "pointsScored",
    # "pointsAllowed",
    "pointsLast3",
    "pointsLast5",
    "runMagnitude",
    "winProbDelta"
]

TARGET = "y_next"

# split = int(len(df_naive_off) * 0.8)

# train_df = df_naive_off.iloc[:split].copy()
# test_df  = df_naive_off.iloc[split:].copy()
# X_train = train_df[FEATURES].fillna(0)
# y_train = train_df[TARGET]
# X_test  = test_df[FEATURES].fillna(0)
# y_test  = test_df[TARGET]

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# model = LogisticRegression(max_iter=2000)
# model.fit(X_train_scaled, y_train)

# y_pred = model.predict(X_test_scaled)
# y_prob = model.predict_proba(X_test_scaled)[:, 1]

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_prob))
# print(classification_report(y_test, y_pred))

# # Feature importance
# importance = pd.Series(model.coef_[0], index=FEATURES)
# print(importance.sort_values(ascending=False))

# groups = df_naive_off["gameId"]

# gkf = GroupKFold(n_splits=5)
# model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)

# scores = []
# X = df_naive_off[FEATURES].fillna(0)
# y = df_naive_off[TARGET]
# for train_idx, test_idx in gkf.split(X, y, groups):
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     scores.append(accuracy_score(y_test, preds))

# print("Mean accuracy:", np.mean(scores))

X = df_naive_off[FEATURES].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100000)
model.fit(X_scaled)   # X is your possession-feature matrix
states = model.predict(X_scaled)
df_naive_off["HMM_State"] = states

game = df_naive_off[df_naive_off["gameId"] == "0022200015"]

plt.figure(figsize=(15,4))
plt.plot(game["HMM_State"], drawstyle="steps-post")
plt.yticks(range(4), ["Cold", "Neutral", "Warm", "Hot"])
plt.title("Momentum State Over Game")
plt.xlabel("Possession")
plt.ylabel("State")
plt.show()