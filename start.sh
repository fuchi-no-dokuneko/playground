#!/usr/bin/env bash
set -euo pipefail

npm run pages:build
npx serve dist -l tcp://0.0.0.0:8082
