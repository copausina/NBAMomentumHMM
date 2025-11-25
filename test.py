from nba_api.stats.endpoints import playbyplay, playbyplayv2, playbyplayv3

v1 = playbyplay.PlayByPlay('0022500247')
df = v1.get_data_frames()[0]
df.to_csv('v1.csv')

# v2 = playbyplayv2.PlayByPlayV2('0022500247')
# df = v2.get_data_frames()[0]
# df.to_csv('v2.csv')

v3 = playbyplayv3.PlayByPlayV3('0022500247')