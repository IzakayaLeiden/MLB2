/**
 * Convert the canonical home-win probability into a neutral, user-facing
 * favorite while retaining both sides of the matchup.
 */
export function predictionDisplay(homeWinProbability) {
  if (!Number.isFinite(homeWinProbability) || homeWinProbability < 0 || homeWinProbability > 1) {
    throw new RangeError("homeWinProbability must be between 0 and 1");
  }

  const awayWinProbability = 1 - homeWinProbability;
  if (homeWinProbability > 0.5) {
    return { favoredSide: "home", favoredProbability: homeWinProbability, homeWinProbability, awayWinProbability };
  }
  if (homeWinProbability < 0.5) {
    return { favoredSide: "away", favoredProbability: awayWinProbability, homeWinProbability, awayWinProbability };
  }
  return { favoredSide: "even", favoredProbability: 0.5, homeWinProbability, awayWinProbability };
}
