export type FeedValidation = {
  available: boolean;
  state: "available" | "empty" | "unavailable";
  code: string | null;
  reason: string | null;
  feed: Record<string, unknown> | null;
};
export function easternDate(value?: Date): string;
export function validatePredictionFeed(payload: unknown, options?: { now?: Date; expectedDate?: string }): FeedValidation;
