"""
Download and merge data from multiple NBA seasons
Supports downloading 2022-23 and 2023-24 seasons
"""
import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playbyplayv3, teamgamelog

def winProbScrape(url):
    """Scrape win probability data from inpredictable.com"""
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    script_text = None

    for script in soup.find_all("script"):
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
    data = json.loads(data_json_str)
    rows = data["rows"]

    records = []
    for r in rows:
        cells = [c.get("v") if c else None for c in r.get("c", [])]
        cells += [None] * (8 - len(cells))
        gt, ann, wprb, tt1, mgn, tt2, lvg, tt3 = cells[:8]

        records.append({
            "gt_seconds": gt,
            "quarter_marker": ann,
            "win_prob_home": wprb,
            "margin": mgn,
            "wp_tooltip_html": tt1,
            "margin_tooltip_html": tt2,
            "leverage": lvg,
            "lev_tooltip_html": tt3,
        })

    df = pd.DataFrame(records)
    df = df[["gt_seconds", "win_prob_home"]].rename(columns={"win_prob_home": "wprb"})
    return df


def parse_clock_iso8601(clock_str):
    """Parse PlayByPlayV3 clock strings like 'PT11M32.00S' into (minutes, seconds)"""
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
    """Convert PlayByPlayV3 ISO clock + period into seconds since game start"""
    mins, secs = parse_clock_iso8601(clock_str)
    if mins is None:
        return None

    remaining = mins * 60 + secs
    elapsed_in_period = period_length - remaining

    return (period - 1) * period_length + elapsed_in_period


def merge(pbp_df: pd.DataFrame, wp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge play-by-play data with win probability data"""
    pbp = pbp_df.copy()
    wp = wp_df.copy()

    pbp["gameId"] = pbp["gameId"].astype(str)
    wp["game_id"] = wp["game_id"].astype(str)

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
        merged = merged.sort_values(["gameId", "t_seconds", "actionId"]).reset_index(drop=True)

    return merged


def download_season(season_str, nba_season_year):
    """
    Download one season's data

    Args:
        season_str: e.g., "2022-23" for file naming
        nba_season_year: e.g., "2022-23" for NBA API
    """
    print(f"\n{'='*60}")
    print(f"Downloading {season_str} season...")
    print(f"{'='*60}\n")

    # Get Milwaukee Bucks team ID
    all_teams = teams.get_teams()
    bucks = [t for t in all_teams if t["full_name"] == "Milwaukee Bucks"][0]
    team_id = bucks["id"]
    print("Bucks Team ID:", team_id)

    # 1. Download Win Probability Data
    print(f"\n[1/3] Downloading win probability data for {season_str}...")
    log = teamgamelog.TeamGameLog(team_id=team_id, season=nba_season_year).get_data_frames()[0]
    log = log.sort_values("GAME_DATE", ascending=True)
    game_list = log["Game_ID"].tolist()

    all_dfs = []
    for i, game_id in enumerate(game_list, 1):
        date_iso = pd.to_datetime(
            log.loc[log["Game_ID"] == game_id, "GAME_DATE"]
        ).dt.strftime("%Y-%m-%d").iloc[0]

        # Determine season year for URL (e.g., 2023 for 2022-23 season)
        url_season = str(int(season_str[:4]) + 1)
        url = f"https://stats.inpredictable.com/nba/wpBox.php?season={url_season}&date={date_iso}&gid={game_id}"

        try:
            df_game = winProbScrape(url)
            df_game["game_id"] = game_id
            df_game["game_date"] = date_iso
            all_dfs.append(df_game)
            print(f"  Downloaded {i}/{len(game_list)}: Game {game_id}")
        except Exception as e:
            print(f"  WARNING: Failed to download game {game_id}: {e}")
            continue

    wp_df = pd.concat(all_dfs, ignore_index=True)
    wp_df["game_date"] = pd.to_datetime(wp_df["game_date"])
    wp_df = wp_df.sort_values(["game_date", "game_id"])

    # 2. Download Play-by-Play Data
    print(f"\n[2/3] Downloading play-by-play data for {season_str}...")
    df_simple = log[["GAME_DATE", "Game_ID", "MATCHUP"]].copy()
    df_simple["OPPONENT"] = df_simple["MATCHUP"].str.split().str[-1]
    df_simple["HOME_GAME"] = df_simple["MATCHUP"].str.contains("vs.", case=False)
    df_simple["GAME_DATE"] = pd.to_datetime(df_simple["GAME_DATE"])
    df_simple = df_simple.sort_values("GAME_DATE").reset_index(drop=True)
    df_simple = df_simple.rename(columns={"Game_ID": "GAME_ID"})
    df_simple = df_simple[["GAME_DATE", "GAME_ID", "OPPONENT", "MATCHUP", "HOME_GAME"]]

    frames = []
    for i, (gid, is_home) in enumerate(zip(df_simple["GAME_ID"], df_simple["HOME_GAME"]), 1):
        v3 = playbyplayv3.PlayByPlayV3(gid)
        v3_df = v3.get_data_frames()[0]
        v3_df["home"] = 'h' if is_home else 'v'
        v3_df["season"] = season_str  # Add season identifier
        frames.append(v3_df)
        print(f"  Downloaded {i}/{len(df_simple)}: Game {gid}")
        time.sleep(1)

    v3season_df = pd.concat(frames, ignore_index=True)
    v3season_df = v3season_df.sort_values(by=["gameId", "actionId"]).reset_index(drop=True)

    # 3. Add time information and merge
    print(f"\n[3/3] Merging play-by-play with win probability data...")
    v3season_df["t_seconds"] = v3season_df.apply(
        lambda r: period_clock_to_seconds(r["period"], r["clock"]),
        axis=1
    )

    v3season_df = merge(v3season_df, wp_df)

    print(f"✓ {season_str} season complete: {len(v3season_df)} records")
    return v3season_df


if __name__ == "__main__":
    # Download both seasons
    seasons = [
        ("2022-23", "2022-23"),
        ("2023-24", "2023-24"),
    ]

    all_seasons = []
    for season_str, nba_season_year in seasons:
        season_df = download_season(season_str, nba_season_year)
        all_seasons.append(season_df)

    # Combine all seasons
    print(f"\n{'='*60}")
    print("Combining all seasons...")
    print(f"{'='*60}\n")

    combined_df = pd.concat(all_seasons, ignore_index=True)
    combined_df = combined_df.sort_values(["season", "gameId", "actionId"]).reset_index(drop=True)

    # Save combined data
    combined_df.to_pickle('v3_multi_season.pkl')
    combined_df.to_csv('v3_multi_season.csv')

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(combined_df):,}")
    print(f"Total games: {combined_df['gameId'].nunique()}")
    print(f"\nRecords by season:")
    print(combined_df.groupby('season').size())
    print(f"\nSaved to: v3_multi_season.pkl and v3_multi_season.csv")
    print(f"{'='*60}\n")
