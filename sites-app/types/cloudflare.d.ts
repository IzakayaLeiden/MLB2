interface Fetcher {
  fetch(request: Request): Promise<Response>;
}

// The starter's optional D1 helper is not used by MLB2; Sites injects the real type when enabled.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type D1Database = any;

declare module "cloudflare:workers" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const env: { DB?: any };
}
