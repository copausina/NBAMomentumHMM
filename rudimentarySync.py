from nba_api.stats.endpoints import playbyplayv3
import pandas as pd

# ---- SET YOUR GAME ID HERE ----
GAME_ID = "0022300075"   # example game ID

# ---- FETCH PLAY-BY-PLAY ----
pbp = playbyplayv3.PlayByPlayV3(game_id=GAME_ID)

# NBA API returns a list of DataFrames; the first one is the PB table
df = pbp.get_data_frames()[0]

# ---------------------------
# ADD PLAY_ID COLUMN
# ---------------------------

play_ids = []
play_counter = 0

n = len(df)

for i in range(n):
    row = df.iloc[i]
    action = row["actionType"]
    play_id = None
    desc = str(row.get("description", "")).lower()

    # 1) Skip subs & timeouts completely
    if action in ["Substitution", "Timeout"]:
        play_ids.append(play_id)
        continue

    # 2) Turnover + next row steal → skip ONLY the turnover
    #    (your data doesn't use 'Steal' as actionType; it puts STEAL in the description)
    if action == "Turnover":
        if i + 1 < n:
            next_desc = str(df.iloc[i + 1]["description"]).lower()
            if "steal" in next_desc:
                play_ids.append(play_id)
                continue

    # 3) Rebound between FT1-of-2 and FT2-of-2 → skip this rebound
    if action == "Rebound":
        if i > 0 and i + 1 < n:
            prev_row = df.iloc[i - 1]
            next_row = df.iloc[i + 1]

            # previous play is a missed FT 1 of 2/3/4
            if prev_row["actionType"] == "Free Throw":      # <-- IMPORTANT: space!
                prev_desc = str(prev_row.get("description", "")).lower()

                is_missed = "miss" in prev_desc
                is_first_of_multi = any(
                    phrase in prev_desc for phrase in ["1 of 2", "1 of 3", "1 of 4"]
                )

                if (
                    is_missed
                    and is_first_of_multi
                #     and next_row["actionType"] == "Free Throw" or next_row["actionType"] == "Substitution"  # <-- also with space
                ):
                    # This rebound is just the board between FT1 and FT2 → no play_id
                    play_ids.append(play_id)   # keep as None
                    continue

    if "off.foul" in desc:
        play_ids.append(play_id)
        continue

    if "instant re" in desc:
        play_ids.append(play_id)
        continue

    if "block" in desc:
        play_ids.append(play_id)
        continue

    if "goaltending" in desc:
        play_ids.append(play_id)
        continue



    # If we didn't continue above, this row gets a play_id
    play_counter += 1
    play_id = play_counter
    play_ids.append(play_id)

df["play_id"] = play_ids
df = df[df["play_id"].notna()].copy()


output_file = f"pbp_v2_{GAME_ID}_with_play_id.csv"

df.to_csv(output_file, index=False)

print(f"Saved PlayByPlayV3 data with play_id to {output_file}")