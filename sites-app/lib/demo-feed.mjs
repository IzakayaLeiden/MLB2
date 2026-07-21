export function demoFeed(targetDate, now = new Date()) {
  const games = [
    [1001, "Chicago White Sox", "CWS", "Davis Martin", "Detroit Tigers", "DET", "Reese Olson", "17:10:00Z", "Comerica Park", 0.63, 1488, 1521, 0.4, 0.7, 1, 1],
    [1002, "Houston Astros", "HOU", "Spencer Arrighetti", "Minnesota Twins", "MIN", "Bailey Ober", "18:10:00Z", "Target Field", 0.613, 1496, 1514, 0.5, 0.6, 2, 1],
    [1003, "San Diego Padres", "SD", "Nick Pivetta", "Chicago Cubs", "CHC", "Matthew Boyd", "18:20:00Z", "Wrigley Field", 0.621, 1502, 1527, 0.6, 0.7, 1, 2],
  ];
  return {
    schema_version: "prediction-feed-v1",
    created_at_utc: now.toISOString(),
    data_through_date: targetDate,
    target_date_et: targetDate,
    timezone: "America/New_York",
    model_version: "model-v1",
    model_sha256: "905a6f001a2f03649cba49027e00ada38c5b3c9bd29e008c953a7c182494bf21",
    quality: { status: "passed", eligible_games: 3, sealed_predictions: 3, late_game_ids: [] },
    previous_file_sha256: null,
    predictions: games.map(([id, awayName, awayAbbr, awayPitcher, homeName, homeAbbr, homePitcher, time, venue, probability, awayElo, homeElo, awayRecent, homeRecent, awayRest, homeRest]) => ({
      game_id: id,
      game_start_utc: `${targetDate}T${time}`,
      official_date: targetDate,
      away_team: { id, name: awayName, abbreviation: awayAbbr, probable_pitcher: awayPitcher },
      home_team: { id: Number(id) + 100, name: homeName, abbreviation: homeAbbr, probable_pitcher: homePitcher },
      venue: { id: Number(id) + 200, name: venue },
      home_win_probability: probability,
      evaluation_eligible: true,
      sealed_before_start_minutes: 300,
      diagnostics: { elo_home_win_probability: probability, away_elo: awayElo, home_elo: homeElo, away_recent_win_pct: awayRecent, home_recent_win_pct: homeRecent, away_rest_days: awayRest, home_rest_days: homeRest, data_through_date: targetDate },
    })),
  };
}
