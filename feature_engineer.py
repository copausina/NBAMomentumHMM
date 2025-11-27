import pandas as pd
import isodate

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
count = 0
actions = set()
for index, row in df.iterrows():
    if row['actionType'] == 'Violation':
        count += 1
        # if count > 20:
        #     break
        nextRow = df.iloc[index+1]
        # actions.add(nextRow['actionType'])
        if nextRow['actionType'] == 'Turnover':
            print(f"{row['gameId']} {row['actionId']} {row['subType']}, Next Row: {nextRow['actionType']} {nextRow['subType']}")
        # print(f"Missed FT ({row['actionId']}) {row['gameId']} {row['period']} {row['clock']}, next row: ({nextRow['actionId']}) {nextRow['gameId']} {nextRow['period']} {nextRow['clock']} {nextRow['actionType']}. {nextRow['description']}")
print(actions)

# remove to run stuff below
exit()

# TODO: handle excessive timeout and too many/few player (also offensive hanging on rim?) technicals, which do end possessions (so final_FTs might not be complete?)
# Only Turnover guarentees end of possession, others depend
possession_ending_actions = {"Made Shot", "Rebound", "Turnover", "Free Throw", "Period"}
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
while i < len(df) and df.iloc[i]['gameId'] == '0022200015':
    cur = df.iloc[i]

    # Beginning of new game
    if cur["actionId"] == 1:
        game_num += 1
        is_home = cur["home"]
        score_diff_start = 0
        clock_start = isodate.parse_duration("PT12M00.00S").total_seconds()
        # skip to after jump ball
        while df.iloc[i]["actionType"] != 'Jump Ball':
            i += 1
        i += 1

    points_scored = 0
    points_allowed = 0
    while(True): 
        match cur["actionType"]:
            case "Free Throw":
                if "MISS" in cur["description"]:
                    pass
                    # TODO: skip "accounting" rebound if applicable
                # Made FT
                else:
                    # Add points to correct team
                    if cur["home"] == is_home:
                        points_scored += 1
                    else:
                        points_allowed += 1
                    # Made final FT ends possession
                    if cur["subType"] in final_FTs:
                        break




    # if cur['actionType'] in possession_ending_actions:

    clock_end = isodate.parse_duration(cur["clock"]).total_seconds()
    duration = clock_start - clock_end
    # home_diff = cur[]
    # new_score_diff = home_diff if is_home else -home_diff
    row = {
        "index": i,
        "gameNum": game_num,
        "gameId": cur["gameId"],
        "isHome": is_home,
        "period": cur["period"],
        "clockStart": clock_start,
        "duration": duration,
        "scoreDiffStart": score_diff_start,
        "pointsScored": points_scored,
        "pointsLast3": idk,
        "pointsLast5": idk,
        "fgPercentageLast3": idk,
        "fgPercentageLast5": idk,
        "leadChange": idk,
        "shotDistance": idk,
        "runMagnitude": idk,
        "orebCount": idk,
        "result": idk        
    }
    rows.append(row)

    # Set starts for next possession
    clock_start = clock_end
    score_diff_start += points_scored

    i += 1

# features_df = pd.DataFrame(rows)