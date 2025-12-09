"""
Basketball-Reference Schedule Scraper

Fetches team schedules and contextual information from basketball-reference.com:
- Team record before each game
- Home/Away status
- Back-to-back games
- Days rest
- Win/loss streaks
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import re

def get_team_schedule(team_abbr, season):
    """
    Fetch team schedule from basketball-reference.com

    Args:
        team_abbr: Team abbreviation (e.g., 'BOS', 'LAL')
        season: Season year (e.g., 2023 for 2022-23 season)

    Returns:
        DataFrame with schedule and context
    """
    # Basketball-reference uses different abbreviations for some teams
    team_mapping = {
        'BKN': 'BRK',  # Brooklyn Nets
        'CHA': 'CHO',  # Charlotte Hornets
        'PHX': 'PHO',  # Phoenix Suns
        'NOP': 'NOP',  # New Orleans Pelicans
    }

    br_team = team_mapping.get(team_abbr, team_abbr)
    url = f"https://www.basketball-reference.com/teams/{br_team}/{season}_games.html"

    print(f"   Fetching {team_abbr} {season-1}-{str(season)[-2:]} season...")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'games'})

        if not table:
            print(f"      ⚠️  Schedule table not found")
            return None

        # Parse table
        rows = table.find('tbody').find_all('tr')
        games = []

        for row in rows:
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class'):
                continue

            cells = row.find_all(['th', 'td'])
            if len(cells) < 7:
                continue

            try:
                game_num = cells[0].text.strip()
                if not game_num or game_num == 'G':
                    continue

                date_str = cells[1].text.strip()
                home_away = cells[3].text.strip()  # @ means away
                opponent = cells[4].text.strip()
                result = cells[5].text.strip()  # W/L
                pts = cells[6].text.strip()
                opp_pts = cells[7].text.strip()

                # Extract W/L and margin
                if result in ['W', 'L']:
                    games.append({
                        'game_num': int(game_num),
                        'date': date_str,
                        'home_away': 'Away' if home_away == '@' else 'Home',
                        'opponent': opponent,
                        'result': result,
                        'pts': int(pts) if pts else None,
                        'opp_pts': int(opp_pts) if opp_pts else None,
                    })

            except (ValueError, IndexError, AttributeError) as e:
                continue

        df = pd.DataFrame(games)

        if len(df) == 0:
            print(f"      ⚠️  No games found")
            return None

        # Calculate contextual features
        df['wins'] = (df['result'] == 'W').astype(int)
        df['losses'] = (df['result'] == 'L').astype(int)

        # Cumulative record BEFORE each game
        df['wins_before'] = df['wins'].shift(1, fill_value=0).cumsum()
        df['losses_before'] = df['losses'].shift(1, fill_value=0).cumsum()
        df['record_before'] = df['wins_before'].astype(str) + '-' + df['losses_before'].astype(str)
        df['win_pct_before'] = df['wins_before'] / (df['wins_before'] + df['losses_before'])
        df['win_pct_before'] = df['win_pct_before'].fillna(0.5)  # First game

        # Win/loss streaks
        df['streak'] = 0
        current_streak = 0
        for i in range(len(df)):
            if i == 0:
                current_streak = 0
            elif df.loc[i-1, 'result'] == df.loc[i, 'result']:
                current_streak += 1
            else:
                current_streak = 0
            df.loc[i, 'streak_before'] = current_streak

        # Days of rest
        df['date_parsed'] = pd.to_datetime(df['date'])
        df['days_rest'] = df['date_parsed'].diff().dt.days - 1
        df['days_rest'] = df['days_rest'].fillna(3)  # First game
        df['back_to_back'] = (df['days_rest'] == 0).astype(int)

        print(f"      ✓ {len(df)} games fetched")
        return df

    except Exception as e:
        print(f"      ✗ Error: {e}")
        return None


def fetch_schedules_for_dataset(possessions_file='possessions_multi_season_with_players.csv'):
    """
    Fetch schedules for all teams in the dataset
    """
    print("=" * 80)
    print("BASKETBALL-REFERENCE SCHEDULE SCRAPER")
    print("=" * 80)

    # Load possessions data
    print("\n1. Loading possessions data...")
    df = pd.read_csv(possessions_file)
    print(f"   Loaded {len(df):,} possessions from {df['gameId'].nunique()} games")

    # Get unique teams
    teams = df['teamTricode'].unique()
    print(f"   Found {len(teams)} unique teams")

    # Determine seasons (2023 = 2022-23, 2024 = 2023-24)
    seasons = [2023, 2024]

    # Fetch schedules
    print("\n2. Fetching schedules from basketball-reference.com...")
    all_schedules = []

    for team in teams:
        for season in seasons:
            schedule = get_team_schedule(team, season)

            if schedule is not None:
                schedule['team'] = team
                schedule['season'] = season
                all_schedules.append(schedule)

            # Be respectful with rate limiting
            time.sleep(3)

    if not all_schedules:
        print("\n   ✗ No schedules fetched")
        return None

    # Combine all schedules
    print("\n3. Combining schedules...")
    combined = pd.concat(all_schedules, ignore_index=True)
    print(f"   ✓ Combined {len(combined):,} games from {combined['team'].nunique()} teams")

    # Save to CSV
    output_file = 'team_schedules_context.csv'
    combined.to_csv(output_file, index=False)
    print(f"\n4. Saved to {output_file}")

    # Display summary
    print("\n" + "=" * 80)
    print("SCHEDULE CONTEXT SUMMARY")
    print("=" * 80)

    print(f"\nContextual Features Added:")
    print(f"   - Record before game (W-L)")
    print(f"   - Win percentage before game")
    print(f"   - Win/loss streak")
    print(f"   - Days of rest")
    print(f"   - Back-to-back indicator")
    print(f"   - Home/Away status")

    print(f"\nSample statistics:")
    print(f"   Avg win pct:     {combined['win_pct_before'].mean():.3f}")
    print(f"   Back-to-backs:   {combined['back_to_back'].sum()} games ({combined['back_to_back'].mean()*100:.1f}%)")
    print(f"   Avg days rest:   {combined['days_rest'].mean():.1f} days")
    print(f"   Home games:      {(combined['home_away']=='Home').sum()} ({(combined['home_away']=='Home').mean()*100:.1f}%)")

    return combined


def merge_schedule_context(possessions_file='possessions_multi_season_with_players.csv',
                           schedules_file='team_schedules_context.csv'):
    """
    Merge schedule context into possessions data
    """
    print("\n" + "=" * 80)
    print("MERGING SCHEDULE CONTEXT")
    print("=" * 80)

    df = pd.read_csv(possessions_file)
    schedules = pd.read_csv(schedules_file)

    print(f"\n   Possessions: {len(df):,}")
    print(f"   Schedule entries: {len(schedules):,}")

    # For merging, we'd need game dates in the possessions data
    # This is a placeholder - actual merging would require game date matching

    print("\n   ⚠️  Note: Merging requires game dates in possessions data")
    print("   Consider adding gameDate field to download_multi_seasons.py")

    output_file = 'possessions_with_schedule_context.csv'
    # df.to_csv(output_file, index=False)  # Uncomment when merge is implemented

    return df


if __name__ == '__main__':
    # Fetch schedules
    schedules = fetch_schedules_for_dataset()

    if schedules is not None:
        print("\n✓ Schedule scraping complete!")
        print("\nNext steps:")
        print("   1. Add game dates to possessions data")
        print("   2. Merge schedule context with possessions")
        print("   3. Re-train models with schedule features")
