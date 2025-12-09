"""
Fetch team names for all games and save to JSON file
Run this once to create game_teams.json
"""
import pandas as pd
import json
from nba_api.stats.endpoints import boxscoretraditionalv2
import time

def get_team_names(game_id):
    """Fetch team names for a game using NBA API"""
    try:
        # Format game_id properly (should be like '0022200015')
        game_id_str = f"00{game_id}"
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id_str)
        team_stats = boxscore.team_stats.get_data_frame()

        if len(team_stats) >= 2:
            return {
                'home': team_stats.iloc[0]['TEAM_ABBREVIATION'],
                'away': team_stats.iloc[1]['TEAM_ABBREVIATION']
            }
    except Exception as e:
        print(f"  ✗ Error for game {game_id}: {e}")

    return None

# Load possessions data
print("Loading possessions data...")
df = pd.read_csv('possessions_multi_season_with_players.csv')
print(f"Found {df['gameId'].nunique()} unique games")

# Fetch team names
print("\nFetching team names from NBA API...")

# Load existing data if available
output_file = 'game_teams.json'
try:
    with open(output_file, 'r') as f:
        game_teams = json.load(f)
    print(f"Loaded {len(game_teams)} existing games from {output_file}")
except FileNotFoundError:
    game_teams = {}
    print("Starting fresh (no existing data)")

for i, game_id in enumerate(df['gameId'].unique()):
    # Skip if already fetched
    if str(game_id) in game_teams and game_teams[str(game_id)]['home'] != 'N/A':
        print(f"  ⊙ [{i+1:3d}/164] Game {game_id}: Already fetched ({game_teams[str(game_id)]['away']} @ {game_teams[str(game_id)]['home']})")
        continue

    teams = get_team_names(game_id)

    if teams:
        game_teams[str(game_id)] = teams
        print(f"  ✓ [{i+1:3d}/164] Game {game_id}: {teams['away']} @ {teams['home']}")
    else:
        game_teams[str(game_id)] = {'home': 'N/A', 'away': 'N/A'}
        print(f"  ✗ [{i+1:3d}/164] Game {game_id}: Failed")

    # Save incrementally after each fetch
    with open(output_file, 'w') as f:
        json.dump(game_teams, f, indent=2)

    # Rate limiting - longer delay to avoid timeouts
    time.sleep(2)

print(f"\n✓ Saved {len(game_teams)} games to {output_file}")
print(f"  Successfully fetched: {sum(1 for v in game_teams.values() if v['home'] != 'N/A')}/{len(game_teams)}")
