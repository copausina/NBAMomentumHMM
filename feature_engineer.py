import pandas as pd
import isodate
from collections import deque

df = pd.read_pickle('v3.pkl')

df.to_csv('check.csv')

# df['actionType'] = df['actionType'].fillna('')
# df['subType'] = df['subType'].fillna('')
# grouped = df.groupby("actionType")["subType"].unique()
# for action, subtypes in grouped.items():
#     print(action)
#     for st in subtypes:
#         print("   -", st)

# Been using this to anaylze specific actionTypes/subTypes
# count = 0
# actions = set()
# for index, row in df.iterrows():
#     if row['actionType'] == 'Violation':
#         count += 1
#         # if count > 20:
#         #     break
#         nextRow = df.iloc[index+1]
#         # actions.add(nextRow['actionType'])
#         if nextRow['actionType'] == 'Turnover':
#             print(f"{row['gameId']} {row['actionId']} {row['subType']}, Next Row: {nextRow['actionType']} {nextRow['subType']}")
#         # print(f"Missed FT ({row['actionId']}) {row['gameId']} {row['period']} {row['clock']}, next row: ({nextRow['actionId']}) {nextRow['gameId']} {nextRow['period']} {nextRow['clock']} {nextRow['actionType']}. {nextRow['description']}")
# print(actions)
# count = 0
# games = []
# for index, row in df.iterrows():
#     if row['actionType'] == 'period' and row['subType'] == 'end':
#         if index+1 == len(df) or df.iloc[index+1]['gameId'] != row['gameId']:
#             count += 1
#             tm = int(row['scoreHome']) if row['home'] == 'h' else int(row['scoreAway'])
#             opp = int(row['scoreAway']) if row['home'] == 'h' else int(row['scoreHome'])
#             game = {'num': count, 'gameId': row['gameId'], 'is_home': row['home'], 'Tm': tm, 'Opp': opp}
#             games.append(game)

# games_df = pd.DataFrame(games)
# games_df.to_csv('games_check.csv', index=False)
# print(games_df['Tm'].sum())
# print(games_df['Opp'].sum())

# remove to run stuff below
# exit()

def max_or_set(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)

def add_points(location, is_home, points, points_scored, points_allowed, run_magnitude):
    # Add points to correct team
    if location == is_home:
        points_scored += points
        run_magnitude = run_magnitude + points if run_magnitude > 0 else points
    else:
        points_allowed += points
        run_magnitude = run_magnitude - points if run_magnitude < 0 else -points
    return points_scored, points_allowed, run_magnitude

# TODO: handle excessive timeout and too many/few player (also offensive hanging on rim?) technicals, which do end possessions (so final_FTs might not be complete?)
# TODO: reviews/challenges, especially for jump balls
# TODO: overtime periods, jump balls
# TODO: exclude garbage time?
# Personal/shooting foul FTs come after any tech FTs
# I am pretty confident flagrant fouls never end possessions too
final_FTs = {"Free Throw 1 of 1", "Free Throw 2 of 2", "Free Throw 3 of 3"}
rows = []
i = 0
game_num = 0
is_home = None
score_diff_start = None
# All of these clock values will be total seconds remaining in period
clock_start = None
run_magnitude = None
points_last_3 = deque(maxlen=3)
points_last_5 = deque(maxlen=5)
home_points = 0
away_points = 0
# These two are really just for debugging
pf_total = 0
pa_total = 0
# and df.iloc[i]['gameId'] == '0022200015'
while i < len(df):
    cur = df.iloc[i]

    points_scored = 0
    points_allowed = 0
    fouls_committed = 0
    fouls_drawn = 0
    shot_distance = None # A possession can have multiple made shots (offensive rebound of and-one FT), take max
    timeouts = 0
    timeouts_opp = 0
    result = None
    o_rebs_grabbed = 0  
    o_rebs_allowed = 0
    
    # Possession loop
    while(True): 
        cur = df.iloc[i]
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
                    # skip to after jump ball
                    while df.iloc[i]["actionType"] != 'Jump Ball':
                        i += 1
                # End of period ends possession
                elif cur['subType'] == 'end':
                    result = 'Period End'
                    # if home_points != int(cur['scoreHome']) or away_points != int(cur['scoreAway']):
                    #     print(f"Mismatch: {cur['gameId']} Period {cur['period']}, computed {home_points}-{away_points}, actual {cur['scoreHome']}-{cur['scoreAway']}")
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
                        # Skip "accounting" rebound
                        i += 1
                # Made FT
                else:
                    if cur["location"] == 'h':
                        home_points += 1
                    else:
                        away_points += 1
                    points_scored, points_allowed, run_magnitude = add_points(cur["location"], is_home, 1, points_scored, points_allowed, run_magnitude)
                    # Made final FT ends possession
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
                points_scored, points_allowed, run_magnitude = add_points(cur["location"], is_home, cur["shotValue"], points_scored, points_allowed, run_magnitude)
                # Made FG ends possession if there was no shooting foul (so if next row is not shooting foul commited by other team)
                if not (df.iloc[i+1]["subType"] == "Shooting" and cur["location"] != df.iloc[i+1]["location"]):
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
                    result = 'Defensive Rebound'
                    break
            case "Turnover":
                if cur["location"] == is_home:
                    result = 'Turnover'
                else:
                    result = 'Opp Turnover'
                # Turnover always ends possession
                break
            case "Timeout":
                if cur["location"] == is_home:
                    timeouts += 1
                else:
                    timeouts_opp += 1
            # Substitution, Violation, Missed Shot, Instant Replay, Ejection, Jump Ball (besides opening tipoff), blank
            case _:
                pass
        i += 1
        

    clock_end = isodate.parse_duration(cur["clock"]).total_seconds()
    duration = clock_start - clock_end
    score_diff_end = score_diff_start + points_scored - points_allowed
    lead_change = (score_diff_start >= 0 and score_diff_end < 0) or (score_diff_start <= 0 and score_diff_end > 0)
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
        # "fgPercentageLast3": idk,
        # "fgPercentageLast5": idk,
        "leadChange": lead_change,
        "shotDistance": shot_distance,
        "runMagnitude": run_magnitude,
        "oRebsGrabbed": o_rebs_grabbed,
        "oRebsAllowed": o_rebs_allowed,
        "timeouts": timeouts,
        "timeoutsOpp": timeouts_opp,
        "result": result
    }
    rows.append(row)

    # Set starts for next possession
    clock_start = clock_end
    score_diff_start = score_diff_end

    i += 1

pf_total += home_points if is_home == 'h' else away_points
pa_total += away_points if is_home == 'h' else home_points
features_df = pd.DataFrame(rows)
features_df.to_csv('possessions.csv', index=False)
print(features_df['pointsScored'].sum())
print(features_df['pointsAllowed'].sum())
print(pf_total)
print(pa_total)
game_counts = features_df['gameNum'].value_counts()
game_counts.to_csv('game_counts.csv', header=['Count'], index=True)