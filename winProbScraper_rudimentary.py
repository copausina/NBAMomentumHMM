import re
from nba_api.stats.endpoints import PlayByPlay
from nba_api.stats.endpoints import teamgamelog
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from nba_api.stats.static import teams


def winProbScrape(url):
    # choose a game
    gameid = "0022300075"   # Bucks @ Sixers (example season 2023-24)

    # # query box score
    PxP = PlayByPlay(game_id=gameid)

    # # Extract player box score as a DataFrame
    PxPdf = PxP.get_data_frames()[0]

    headers = {
        "User-Agent": "Mozilla/5.0"  # helps avoid some sites blocking the request
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    script_text = None

    for script in soup.find_all("script"):
        # script.string is the JS source as a Python string, or None for external scripts
        if script.string and "var data_wprb" in script.string:
            script_text = script.string
            break

    if script_text is None:
        raise RuntimeError("Couldn't find a <script> block with 'var data_wprb' in it.")


    pattern = re.compile(
        r"var\s+data_wprb\s*=\s*new google\.visualization\.DataTable\((\{.*?\})\);",
        re.DOTALL
    )
    match = pattern.search(script_text)

    if not match:
        raise RuntimeError("Found the script, but couldn't extract data_wprb JSON.")

    data_json_str = match.group(1)

    # --- 5. Parse the JSON into a Python dict ---
    data = json.loads(data_json_str)

    # data["cols"] = column definitions
    # data["rows"] = actual data

    rows = data["rows"]

    records = []
    for r in rows:
        cells = [c.get("v") if c else None for c in r.get("c", [])]
        # pad to 8 elements just in case
        cells += [None] * (8 - len(cells))
        gt, ann, wprb, tt1, mgn, tt2, lvg, tt3 = cells[:8]

        records.append(
            {
                "gt_seconds": gt,
                "quarter_marker": ann,      # 'Q1', 'Q2', etc, or None
                "win_prob_home": wprb,      # 0.0–1.0
                "margin": mgn,
                "wp_tooltip_html": tt1,
                "margin_tooltip_html": tt2,
                "leverage": lvg,
                "lev_tooltip_html": tt3,
            }
        )

    # df = pd.DataFrame(records)
    # df = pd.DataFrame({
    #     "play_id": range(1, len(df) + 1),
    #     "wprb": df["win_prob_home"]
    # })
    
    df = pd.DataFrame(records)
    df = df[["gt_seconds", "win_prob_home"]].rename(columns={"win_prob_home": "wprb"})
    return df

#-------------------------------------------------------------------------
# Loop through every bucks game, export csv

# Pull bucks TeamID
all_teams = teams.get_teams()
bucks = [t for t in all_teams if t["full_name"] == "Milwaukee Bucks"][0]
team_id = bucks["id"]

# Game log
log = teamgamelog.TeamGameLog(team_id=team_id, season="2022-23").get_data_frames()[0]

# get games in order
log = log.sort_values("GAME_DATE", ascending=True)

# Turn game log into actionable list
game_list = log["Game_ID"].tolist()

# Loop through game list, make one giant csv
all_dfs = []
for game_id in game_list:
    date_iso = pd.to_datetime(
        log.loc[log["Game_ID"] == game_id, "GAME_DATE"]
    ).dt.strftime("%Y-%m-%d").iloc[0]

    url = f"https://stats.inpredictable.com/nba/wpBox.php?season=2023&date={date_iso}&gid={game_id}"

    df_game = winProbScrape(url)   # ← Your function
    df_game["game_id"] = game_id
    df_game["game_date"] = date_iso

    all_dfs.append(df_game)

full = pd.concat(all_dfs, ignore_index=True)

full["game_date"] = pd.to_datetime(full["game_date"])
full = full.sort_values(["game_date", "game_id"])

full.to_csv("bucks_winprob_2023.csv", index=False)
full.to_pickle('wp.pkl')
