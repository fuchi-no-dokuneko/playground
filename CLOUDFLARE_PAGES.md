# Cloudflare Pages

This app is built as a fully static site. All data generation, model training,
visualization, sweeps, and smoke checks run in the browser or local Node tools.

## Build Settings

- Repository: `fuchi-no-dokuneko/playground`
- Production branch: `master`
- Build command: `npm run pages:build`
- Output directory: `dist`
- Wrangler output config: `pages_build_output_dir = "dist"`
- Cloudflare project: `tensor-playground-research`

Cloudflare Pages can deploy this project because the build produces a static
`index.html`, JavaScript, CSS, and browser assets in `dist`; the app does not
require an application server.

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

## Automated Deployment

The `Build and deploy pages` GitHub Actions workflow builds and smoke-tests each
push and pull request. Pushes to `master` deploy the validated `dist` artifact
to both hosting providers.

GitHub Pages uses GitHub Actions directly and does not require a repository
secret. Before the first deployment, enable it under **Settings > Pages** and
select **GitHub Actions** as the source. Cloudflare deployment requires these
GitHub repository secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

The token must be able to edit Cloudflare Pages projects for the account.

## Direct Deployment

Alternatively, authenticate Wrangler and deploy the built directory directly:

```bash
npm run pages:build
npx wrangler login
npm run pages:deploy
```

The workflow and direct command both deploy to the
`tensor-playground-research` Pages project.
