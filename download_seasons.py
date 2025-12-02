import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playbyplay, playbyplayv3, teamgamelog
import pandas as pd
import pickle
import re

# takes in win prob
wp_df = pd.read_pickle("wp.pkl")

# play_id allows us to append win prob file to existing file
# def assign_play_ids(df):
#     play_ids = []
#     play_counter = 0
#     n = len(df)

#     for i in range(n):
#         row = df.iloc[i]
#         action = row["actionType"]
#         sub_type = row["subType"]
#         play_id = None
#         desc = str(row.get("description", "")).lower()

#         # 1) Skip subs & timeouts completely
#         if action in ["Substitution", "Timeout"]:
#             play_ids.append(play_id)
#             continue

#         # 2) skip steals
#         if action == "Steal":
#             play_ids.append(None)
#             continue

#         # 3) eliminate fake rebound after missed fg
#         if action == "Rebound":
#             if i > 0 and i + 1 < n:
#                 prev_row = df.iloc[i - 1]
#                 prev_desc = str(prev_row.get("description", "")).lower()

#                 if prev_row["actionType"] == "Free Throw":
#                     is_missed = "miss" in prev_desc
#                     is_first_of_multi = any(
#                         phrase in prev_desc for phrase in ["1 of 2", "1 of 3", "2 of 3", "free throw technical"]
#                     )

#                     if is_missed and is_first_of_multi:
#                         play_ids.append(play_id)
#                         continue
            
#             # phantom rebound after flagrant free throws
#             if i > 0:
#                 prev_desc = str(df.iloc[i - 1].get("description", "")).lower()

#                 if "flagrant" in prev_desc and ("1 of 1" in prev_desc or "2 of 2" in prev_desc):
#                     play_ids.append(None)
#                     continue

#         # 4) Keywords to ignore
#         if any(kw in desc for kw in ["off.foul", "instant re", "block", "defensive goaltending", "steal", "charge foul", "ejection", "double technical", "free throw flagrant 1 of 1"]):
#             play_ids.append(play_id)
#             continue

#         if sub_type == "Lane":
#             play_ids.append(None)
#             continue

#         if sub_type == "Delay Technical":
#             play_ids.append(None)
#             continue

#         if sub_type == "Free Throw Technical":
#             if i > 0 and df.iloc[i - 1].get("subType", "") == "Delay Technical":
#                 play_ids.append(None)   # skip this tech FT
#                 continue

#         # Otherwise: actual play
#         play_counter += 1
#         play_ids.append(play_counter)

#     df["play_id"] = play_ids
#     return df

def parse_clock_iso8601(clock_str):
    """
    Parses PlayByPlayV3 clock strings like 'PT11M32.00S' into (minutes, seconds).
    Returns minutes (int), seconds (float).
    """
    if pd.isna(clock_str):
        return None, None

    match = re.match(r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", clock_str)
    if not match:
        return None, None

    mins = match.group(1)
    secs = match.group(2)

    mins = int(mins) if mins is not None else 0
    secs = float(secs) if secs is not None else 0.0

    return mins, secs


def period_clock_to_seconds(period, clock_str, period_length=12*60):
    """
    Converts PlayByPlayV3 ISO clock + period into seconds since game start.
    Example: period 1, clock 'PT11M32.00S'
    """
    mins, secs = parse_clock_iso8601(clock_str)
    if mins is None:
        return None

    remaining = mins * 60 + secs          # time left in period
    elapsed_in_period = period_length - remaining

    return (period - 1) * period_length + elapsed_in_period

def merge(pbp_df: pd.DataFrame, wp_df: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp_df.copy()
    wp = wp_df.copy()

    # Make sure game IDs match type
    pbp["gameId"] = pbp["gameId"].astype(str)
    wp["game_id"] = wp["game_id"].astype(str)

    # Keep only what we need from win-prob
    if "gt_seconds" not in wp.columns:
        raise KeyError(f"win-prob df is missing 'gt_seconds'. Columns: {wp.columns.tolist()}")
    if "wprb" not in wp.columns:
        raise KeyError(f"win-prob df is missing 'wprb'. Columns: {wp.columns.tolist()}")

    wp = wp[["game_id", "gt_seconds", "wprb"]]

    merged_games = []

    for gid, pbp_g in pbp.groupby("gameId"):
        wp_g = wp[wp["game_id"] == gid]

        if wp_g.empty:
            merged_games.append(pbp_g.copy())
            continue

        pbp_g = pbp_g.copy()
        wp_g = wp_g.copy()

        pbp_g["t_seconds"] = pbp_g["t_seconds"].astype(float)
        wp_g["gt_seconds"] = wp_g["gt_seconds"].astype(float)

        pbp_g = pbp_g.sort_values("t_seconds")
        wp_g = wp_g.sort_values("gt_seconds")

        merged_g = pd.merge_asof(
            pbp_g,
            wp_g,
            left_on="t_seconds",
            right_on="gt_seconds",
            direction="backward",
        )

        merged_games.append(merged_g)

    merged = pd.concat(merged_games, ignore_index=True)
    if "actionNumber" in merged.columns:
        merged = merged.sort_values(["gameId", "t_seconds", "actionNumber"]).reset_index(drop=True)
    else:
        # fallback if you prefer actionId
        merged = merged.sort_values(["gameId", "t_seconds", "actionId"]).reset_index(drop=True)

    return merged

# MAIN------------------------------------------------
# 1. Get Milwaukee Bucks team ID
all_teams = teams.get_teams()
bucks = [t for t in all_teams if t["full_name"] == "Milwaukee Bucks"][0]
team_id = bucks["id"]
print("Bucks Team ID:", team_id)

# 2. Pull their 2022-23 regular season game log
log = teamgamelog.TeamGameLog(
    team_id=team_id,
    season="2022-23",                 # NBA 2023 season = 2023-24
    season_type_all_star="Regular Season"
)

df = log.get_data_frames()[0]
# print("Columns:", df.columns.tolist())  # sanity check

# 3. Build a simple table: date, game id, opponent, matchup
df_simple = df[["GAME_DATE", "Game_ID", "MATCHUP"]].copy()

# Parse opponent from MATCHUP (e.g., 'MIL vs. BOS' or 'MIL @ BOS')
df_simple["OPPONENT"] = df_simple["MATCHUP"].str.split().str[-1]
df_simple["HOME_GAME"] = df_simple["MATCHUP"].str.contains("vs.", case=False)

# Make GAME_DATE a real datetime and sort
df_simple["GAME_DATE"] = pd.to_datetime(df_simple["GAME_DATE"])
df_simple = df_simple.sort_values("GAME_DATE").reset_index(drop=True)

# Optional: rename Game_ID -> GAME_ID for consistency
df_simple = df_simple.rename(columns={"Game_ID": "GAME_ID"})

# Reorder columns
df_simple = df_simple[["GAME_DATE", "GAME_ID", "OPPONENT", "MATCHUP", "HOME_GAME"]]

df_simple.to_csv('bucks_2022_23_games.csv', index=False)

frames = []
for gid, is_home in zip(df_simple["GAME_ID"], df_simple["HOME_GAME"]):
    v3 = playbyplayv3.PlayByPlayV3(gid)
    v3_df = v3.get_data_frames()[0]
    v3_df["home"] = 'h' if is_home else 'v'
    frames.append(v3_df)
    time.sleep(1)

# add play_id


v3season_df = pd.concat(frames, ignore_index=True)
# Need to sort by gameId first bc the first event of game 1 has the same index as the first event of game 2, 3, etc.
v3season_df = v3season_df.sort_values(by=["gameId", "actionId"]).reset_index(drop=True)

# assign play id
# v3season_df = assign_play_ids(v3season_df)
v3season_df["t_seconds"] = v3season_df.apply(
    lambda r: period_clock_to_seconds(r["period"], r["clock"]),
    axis=1
)

# call merge
v3season_df = merge(v3season_df, wp_df)

# pickle so you don't have to hit api again
v3season_df.to_pickle('v3.pkl')
v3season_df.to_csv('v3.csv')
