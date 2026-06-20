# Cloudflare Pages

This app is built as a fully static site. All data generation, model training,
visualization, sweeps, and smoke checks run in the browser or local Node tools.

## Build Settings

- Build command: `npm run pages:build`
- Output directory: `dist`
- Wrangler output config: `pages_build_output_dir = "dist"`

## Local Verification

```bash
npm run pages:build
npm run serve
npm run smoke:render
npm run smoke:selenium:firefox
```

`npm run smoke:render` expects the static server to be reachable at
`http://127.0.0.1:4173/` unless `PLAYGROUND_URL` is set.

`npm run smoke:selenium:firefox` runs Selenium WebDriver against headless
Firefox. It uses `FIREFOX_BINARY` when set, otherwise it uses the local
Playwright Firefox binary installed at
`/home/vmadmin/.cache/ms-playwright/firefox-1532/firefox/firefox`.

## GitHub Fork Step

The local branch is `hyper0cube-random-walk`. The actual GitHub fork must be
created by an authenticated GitHub user, then the local remote can be updated,
for example:

```bash
git remote set-url origin https://github.com/<your-account>/playground.git
git push -u origin hyper0cube-random-walk
```
