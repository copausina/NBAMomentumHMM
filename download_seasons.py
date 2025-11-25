import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playbyplay, playbyplayv3, teamgamelog
import pandas as pd
import pickle

# 1. Get Milwaukee Bucks team ID
all_teams = teams.get_teams()
bucks = [t for t in all_teams if t["full_name"] == "Milwaukee Bucks"][0]
team_id = bucks["id"]
print("Bucks Team ID:", team_id)

# 2. Pull their 2023-24 regular season game log
log = teamgamelog.TeamGameLog(
    team_id=team_id,
    season="2023-24",                 # NBA 2023 season = 2023-24
    season_type_all_star="Regular Season"
)

df = log.get_data_frames()[0]
print("Columns:", df.columns.tolist())  # sanity check

# 3. Build a simple table: date, game id, opponent, matchup
df_simple = df[["GAME_DATE", "Game_ID", "MATCHUP"]].copy()

# Parse opponent from MATCHUP (e.g., 'MIL vs. BOS' or 'MIL @ BOS')
df_simple["OPPONENT"] = df_simple["MATCHUP"].str.split().str[-1]

# Make GAME_DATE a real datetime and sort
df_simple["GAME_DATE"] = pd.to_datetime(df_simple["GAME_DATE"])
df_simple = df_simple.sort_values("GAME_DATE").reset_index(drop=True)

# Optional: rename Game_ID -> GAME_ID for consistency
df_simple = df_simple.rename(columns={"Game_ID": "GAME_ID"})

# Reorder columns
df_simple = df_simple[["GAME_DATE", "GAME_ID", "OPPONENT", "MATCHUP"]]

df_simple

v1season_df = pd.DataFrame()
v3season_df = pd.DataFrame()
# Loop thru game ids, downloading pbp data (v1 and v2 format) and appending it to one big df
for gid in df_simple["GAME_ID"].unique():
    v1 = playbyplay.PlayByPlay(gid)
    v1_df = v1.get_data_frames()[0]
    v3 = playbyplayv3.PlayByPlayV3(gid)
    v3_df = v1.get_data_frames()[0]
    v1season_df = pd.concat([v1season_df, v1_df])
    v3season_df = pd.concat([v3season_df, v3_df])
    # prevents read timeouts
    time.sleep(1)

# pickle so you don't have to hit api again
v1season_df.to_pickle('v1.pkl')
v3season_df.to_pickle('v3.pkl')


# v1 = playbyplay.PlayByPlay('0022500247')
# df = v1.get_data_frames()[0]
# df.to_csv('v1.csv')

# v2 = playbyplayv2.PlayByPlayV2('0022500247')
# df = v2.get_data_frames()[0]
# df.to_csv('v2.csv')

# v3 = playbyplayv3.PlayByPlayV3('0022500247')
# df = v2.get_data_frames()[0]
# df.to_csv('v3.csv')