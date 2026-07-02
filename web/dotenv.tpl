# Local dev: leave VITE_API_URL empty and rely on Vite's dev-server proxy
# (see vite.config.ts) which forwards /api/* to http://localhost:9627.
#
# For a production build that runs on GitHub Pages, VITE_API_URL must point
# to the public backend URL (e.g. Render). That value is injected at build
# time by the Pages workflow via the VITE_API_URL repository variable.

VITE_BASE_URL=./
VITE_API_URL=

VITE_SUPABASE_URL=
VITE_SUPABASE_PUBLISHABLE_KEY=
