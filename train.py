import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

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
]

TARGET = "y_next"

print(df_naive_off[["gameId", "pointsScored", "y_next"]].head(10))

split = int(len(df_naive_off) * 0.8)

train_df = df_naive_off.iloc[:split].copy()
test_df  = df_naive_off.iloc[split:].copy()
X_train = train_df[FEATURES].fillna(0)
y_train = train_df[TARGET]
X_test  = test_df[FEATURES].fillna(0)
y_test  = test_df[TARGET]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.Series(model.coef_[0], index=FEATURES)
print(importance.sort_values(ascending=False))