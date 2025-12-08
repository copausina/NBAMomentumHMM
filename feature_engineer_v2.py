"""
Enhanced Feature Engineering with Outlier Filtering
Extracts possession-level features from play-by-play data
"""
import pandas as pd
import isodate
import re
from collections import deque

def is_outlier_possession(df, i):
    """
    Identify outlier possessions that should be excluded from analysis

    Outliers include:
    1. Intentional fouls (typically at end of game)
    2. End of period heaves (desperate long shots)
    3. Technical fouls that don't reflect normal gameplay

    Args:
        df: DataFrame with play-by-play data
        i: Current index

    Returns:
        bool: True if possession should be excluded
    """
    cur = df.iloc[i]

    # Parse clock to seconds
    def parse_clock(clock_str):
        if pd.isna(clock_str):
            return None
        match = re.match(r'PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?', str(clock_str))
        if not match:
            return None
        mins = int(match.group(1)) if match.group(1) else 0
        secs = float(match.group(2)) if match.group(2) else 0
        return mins * 60 + secs

    clock_seconds = parse_clock(cur.get('clock'))
    description = str(cur.get('description', '')).lower()
    action_type = str(cur.get('actionType', ''))
    sub_type = str(cur.get('subType', ''))

    # 1. Intentional fouls (usually has "intentional" in description)
    if 'intentional' in description:
        return True

    # 2. Clear path fouls (not typical gameplay)
    if 'clear path' in description:
        return True

    # 3. End of period heaves (last 5 seconds, shot > 30 feet)
    if clock_seconds is not None and clock_seconds <= 5:
        shot_distance = cur.get('shotDistance')
        if pd.notna(shot_distance) and shot_distance >= 30:
            return True

    # 4. Away from play fouls (not related to possession outcome)
    if 'away from play' in description:
        return True

    # 5. Delay of game technicals (administrative, not gameplay)
    if sub_type == 'Delay Technical':
        return True

    return False


def max_or_set(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def add_points(location, is_home, points, points_scored, points_allowed, run_magnitude):
    """Add points to correct team"""
    if location == is_home:
        points_scored += points
        run_magnitude = run_magnitude + points if run_magnitude > 0 else points
    else:
        points_allowed += points
        run_magnitude = run_magnitude - points if run_magnitude < 0 else -points
    return points_scored, points_allowed, run_magnitude


def extract_features(input_file, output_file):
    """
    Extract possession-level features from play-by-play data

    Args:
        input_file: Path to v3.pkl or v3_multi_season.pkl
        output_file: Path to save possessions.csv
    """
    df = pd.read_pickle(input_file)

    print(f"Loaded {len(df):,} records from {input_file}")
    print(f"Unique games: {df['gameId'].nunique()}")
    if 'season' in df.columns:
        print(f"Seasons: {df['season'].unique()}")

    # Identify and mark outliers
    print("\nIdentifying outlier possessions...")
    outlier_flags = []
    for i in range(len(df)):
        outlier_flags.append(is_outlier_possession(df, i))

    outlier_count = sum(outlier_flags)
    print(f"Found {outlier_count} outlier records ({outlier_count/len(df)*100:.2f}%)")

    # Final FT types that end possession
    final_FTs = {"Free Throw 1 of 1", "Free Throw 2 of 2", "Free Throw 3 of 3"}

    rows = []
    i = 0
    game_num = 0
    is_home = None
    score_diff_start = None
    clock_start = None
    run_magnitude = None
    points_last_3 = deque(maxlen=3)
    points_last_5 = deque(maxlen=5)
    home_points = 0
    away_points = 0
    pf_total = 0
    pa_total = 0
    outliers_skipped = 0

    while i < len(df):
        cur = df.iloc[i]

        # Skip outliers during possession processing
        if outlier_flags[i] and cur["actionType"] not in ["period"]:
            outliers_skipped += 1
            i += 1
            continue

        points_scored = 0
        points_allowed = 0
        fouls_committed = 0
        fouls_drawn = 0
        shot_distance = None
        timeouts = 0
        timeouts_opp = 0
        result = None
        o_rebs_grabbed = 0
        o_rebs_allowed = 0

        # Possession loop
        while True:
            cur = df.iloc[i]

            # Skip outliers within possession
            if outlier_flags[i] and cur["actionType"] not in ["period"]:
                outliers_skipped += 1
                i += 1
                if i >= len(df):
                    break
                continue

            match cur["actionType"]:
                case "period":
                    # Beginning of new game
                    if cur['subType'] == 'start' and cur['period'] == 1:
                        game_num += 1
                        is_home = cur["home"]
                        score_diff_start = 0
                        clock_start = isodate.parse_duration("PT12M00.00S").total_seconds()
                        run_magnitude = 0
                        points_last_3.clear()
                        points_last_5.clear()
                        home_points = 0
                        away_points = 0
                        # Skip to after jump ball
                        while df.iloc[i]["actionType"] != 'Jump Ball':
                            i += 1
                    # End of period ends possession
                    elif cur['subType'] == 'end':
                        result = 'Period End'
                        if i+1 < len(df) and df.iloc[i+1]['gameId'] != cur['gameId']:
                            pf_total += home_points if is_home == 'h' else away_points
                            pa_total += away_points if is_home == 'h' else home_points
                        break
                    # Start of period but not of game
                    else:
                        clock_start = isodate.parse_duration("PT12M00.00S").total_seconds()

                case "Free Throw":
                    if "MISS" in cur["description"]:
                        if cur["subType"] not in final_FTs:
                            i += 1  # Skip accounting rebound
                    else:
                        if cur["location"] == 'h':
                            home_points += 1
                        else:
                            away_points += 1
                        points_scored, points_allowed, run_magnitude = add_points(
                            cur["location"], is_home, 1, points_scored, points_allowed, run_magnitude
                        )
                        if cur["subType"] in final_FTs:
                            result = 'Made FT' if cur["location"] == is_home else 'Opp Made FT'
                            break

                case "Foul":
                    if cur["location"] == is_home:
                        fouls_committed += 1
                    else:
                        fouls_drawn += 1

                case "Made Shot":
                    shot_distance = max_or_set(shot_distance, cur["shotDistance"])
                    if cur["location"] == 'h':
                        home_points += cur["shotValue"]
                    else:
                        away_points += cur["shotValue"]
                    points_scored, points_allowed, run_magnitude = add_points(
                        cur["location"], is_home, cur["shotValue"],
                        points_scored, points_allowed, run_magnitude
                    )
                    if not (df.iloc[i+1]["subType"] == "Shooting" and
                           cur["location"] != df.iloc[i+1]["location"]):
                        result = 'Made FG' if cur["location"] == is_home else 'Opp Made FG'
                        break

                case "Rebound":
                    # Offensive rebound
                    if df.iloc[i-1]["location"] == cur["location"]:
                        if cur["location"] == is_home:
                            o_rebs_grabbed += 1
                        else:
                            o_rebs_allowed += 1
                    # Defensive rebound ends possession
                    else:
                        result = 'Defensive Rebound' if cur["location"] == is_home else 'Opp Defensive Rebound'
                        break

                case "Turnover":
                    if cur["location"] == is_home:
                        result = 'Turnover'
                    else:
                        result = 'Opp Turnover'
                    break

                case "Timeout":
                    if cur["location"] == is_home:
                        timeouts += 1
                    else:
                        timeouts_opp += 1

                case "Missed Shot":
                    shot_distance = max_or_set(shot_distance, cur["shotDistance"])

                case _:
                    pass

            i += 1
            if i >= len(df):
                break

        if i >= len(df):
            break

        clock_end = isodate.parse_duration(cur["clock"]).total_seconds()
        duration = clock_start - clock_end
        score_diff_end = score_diff_start + points_scored - points_allowed
        lead_change = (score_diff_start >= 0 and score_diff_end < 0) or \
                     (score_diff_start <= 0 and score_diff_end > 0)
        points_last_3.append(points_scored)
        points_last_5.append(points_scored)

        row = {
            "index": i,
            "gameNum": game_num,
            "gameId": cur["gameId"],
            "isHome": is_home,
            "period": cur["period"],
            "clockStart": clock_start,
            "duration": duration,
            "scoreHome": home_points,
            "scoreAway": away_points,
            "scoreDiffStart": score_diff_start,
            "pointsScored": points_scored,
            "pointsAllowed": points_allowed,
            "foulsCommitted": fouls_committed,
            "foulsDrawn": fouls_drawn,
            "pointsLast3": sum(points_last_3),
            "pointsLast5": sum(points_last_5),
            "leadChange": lead_change,
            "shotDistance": shot_distance,
            "runMagnitude": run_magnitude,
            "oRebsGrabbed": o_rebs_grabbed,
            "oRebsAllowed": o_rebs_allowed,
            "timeouts": timeouts,
            "timeoutsOpp": timeouts_opp,
            "result": result
        }

        # Add season if available
        if 'season' in cur:
            row['season'] = cur['season']

        rows.append(row)

        # Set starts for next possession
        clock_start = clock_end
        score_diff_start = score_diff_end

        i += 1

    pf_total += home_points if is_home == 'h' else away_points
    pa_total += away_points if is_home == 'h' else home_points

    features_df = pd.DataFrame(rows)
    features_df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total possessions: {len(features_df):,}")
    print(f"Outliers skipped: {outliers_skipped}")
    print(f"Points scored: {features_df['pointsScored'].sum():,}")
    print(f"Points allowed: {features_df['pointsAllowed'].sum():,}")
    print(f"Validation - PF total: {pf_total:,}")
    print(f"Validation - PA total: {pa_total:,}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")

    return features_df


if __name__ == "__main__":
    import sys

    # Check if multi-season file exists
    import os
    if os.path.exists('v3_multi_season.pkl'):
        print("Using multi-season data (v3_multi_season.pkl)")
        input_file = 'v3_multi_season.pkl'
        output_file = 'possessions_multi_season.csv'
    else:
        print("Using single season data (v3.pkl)")
        input_file = 'v3.pkl'
        output_file = 'possessions.csv'

    extract_features(input_file, output_file)
