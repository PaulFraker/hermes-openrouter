# Hermes OpenRouter Sandbox

Isolated Hermes Agent Docker setup with persistent memory/storage, Telegram gateway credentials, and automatic free OpenRouter model selection.

## Files

- `.env` — local credentials and selector settings. Ignored by git.
- `.env.example` — sanitized credential/settings template.
- `hermes-config.example.yaml` — sanitized Hermes OpenRouter config template for `hermes-home/config.yaml`.
- `hermes-home/` — persistent `HERMES_HOME` mounted as `/opt/data`; contains config, memory, sessions, logs. Ignored by git.
- `workspace/` — safe local workspace mounted as `/workspace`.
- `selector/openrouter_free_selector.py` — free-model selector and optional benchmark.
- `reports/` — model selection reports; JSON/log output is ignored by git.

## Required credentials

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=123456789:ABC...
TELEGRAM_ALLOWED_USERS=1560734470
TELEGRAM_HOME_CHANNEL=1560734470
```

`TELEGRAM_ALLOWED_USERS` is strongly recommended so only listed Telegram user IDs can talk to this sandbox bot.

Hermes itself reads secrets from its mounted `HERMES_HOME` env file too:

```text
hermes-home/.env  -> mounted as /opt/data/.env
```

Bootstrap a fresh runtime config with:

```bash
mkdir -p hermes-home reports workspace
cp .env.example .env
cp hermes-config.example.yaml hermes-home/config.yaml
cp .env hermes-home/.env
```

Keep `.env` and `hermes-home/.env` in sync when rotating credentials. Compose uses `.env`; Hermes CLI/gateway resolves provider credentials via `/opt/data/.env`.

## Start

```bash
cd /opt/apps/hermes-openrouter
docker compose --env-file .env up -d --build
```

The gateway starts immediately with the current `model.default`.
The `model-selector` service refreshes selection every `SELECTOR_INTERVAL_SECONDS` and applies benchmarked free-model changes.

## Check status

```bash
docker compose ps
docker compose logs -f hermes-gateway
docker compose logs -f model-selector
cat reports/model-selection.json
```

## Manual selector run

Report only:

```bash
docker compose exec hermes-gateway python3 /selector/openrouter_free_selector.py \
  --config /opt/data/config.yaml \
  --report /reports/manual-report.json \
  --mode report_only
```

Apply:

```bash
docker compose exec hermes-gateway python3 /selector/openrouter_free_selector.py \
  --config /opt/data/config.yaml \
  --report /reports/manual-apply.json \
  --mode apply
```

## Manual Hermes test

```bash
docker compose exec hermes-gateway bash -c 'hermes chat -q \
  "Antworte exakt: OK"'
```

## Benchmark mode

Enabled in this sandbox because the highest-scored free models can be temporarily upstream-rate-limited. The selector live-tests the top free candidates before applying a switch:

```env
SELECTOR_ENABLE_BENCHMARK=1
SELECTOR_MAX_BENCHMARK_MODELS=6
```

Benchmark remains free-only, but OpenRouter free models can still be rate-limited. If this becomes too noisy, set `SELECTOR_ENABLE_BENCHMARK=0`.

## Stop

```bash
docker compose down
```

Persistent state remains in `hermes-home/`, `workspace/`, and `reports/`.
