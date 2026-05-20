#!/usr/bin/env python3
"""Select the best currently-free OpenRouter model for a Hermes sandbox.

Modes:
- report_only: write /reports/model-selection.json, do not change config.
- apply: update model.default in Hermes config if switch conditions are met.

The selector is deliberately free-only. It never chooses a paid model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
UTC = dt.timezone.utc

CODING_KEYWORDS = (
    "code", "coding", "programming", "developer", "software", "agentic",
    "reasoning", "debug", "bug", "python", "javascript", "typescript",
    "rust", "shell", "bash", "sql", "math", "logic",
)
NEGATIVE_KEYWORDS = (
    "image", "vision only", "audio", "roleplay", "uncensored", "story", "creative writing",
)

BENCHMARK_PROMPTS = [
    {
        "name": "python_dedup",
        "prompt": "Return only Python code. Implement dedupe_keep_order(items) that removes duplicates while preserving order and works for unhashable dict/list values too.",
        "checks": ["def dedupe_keep_order", "return", "for"],
    },
    {
        "name": "json_exact",
        "prompt": "Return exactly one compact JSON object and nothing else: keys ok(boolean), language(string), files(array). Values: ok=true, language=python, files=[\"main.py\"].",
        "checks": ["{", "ok", "language", "files"],
    },
    {
        "name": "patch_reasoning",
        "prompt": "A test fails because a function mutates its input list. In max 80 words, explain the safest fix for a production codebase.",
        "checks": ["copy", "input"],
    },
]


def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def parse_expiration(value: Any) -> dt.datetime | None:
    if value in (None, "", 0, "0"):
        return None
    if isinstance(value, (int, float)):
        # OpenRouter timestamps are unix seconds when present.
        return dt.datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            return dt.datetime.fromtimestamp(float(s), tz=UTC)
        try:
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
    return None


def pricing_is_zero(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    def zero(key: str) -> bool:
        try:
            return float(pricing.get(key, 1)) == 0.0
        except (TypeError, ValueError):
            return False
    return zero("prompt") and zero("completion")


def fetch_json(url: str, api_key: str | None = None, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None
    headers = {
        "Accept": "application/json",
        "HTTP-Referer": "https://local.hermes-openrouter-sandbox",
        "X-Title": "Hermes OpenRouter Free Model Sandbox",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def score_model(model: dict[str, Any], safety_deadline: dt.datetime, min_context: int) -> tuple[float, list[str]]:
    reasons: list[str] = []
    model_id = str(model.get("id") or "")
    name = str(model.get("name") or "")
    description = str(model.get("description") or "")
    text = f"{model_id} {name} {description}".lower()
    supported = set(model.get("supported_parameters") or [])
    arch = model.get("architecture") or {}
    input_modalities = set(arch.get("input_modalities") or [])
    output_modalities = set(arch.get("output_modalities") or [])
    context = int(model.get("context_length") or 0)
    expiration = parse_expiration(model.get("expiration_date"))

    score = 0.0
    score += 4.0
    reasons.append("zero prompt/completion pricing")

    if model_id.endswith(":free"):
        score += 1.5
        reasons.append(":free id")
    elif pricing_is_zero(model):
        score += 0.75
        reasons.append("promotional zero pricing")

    if expiration is None:
        score += 1.0
        reasons.append("no published expiration")
    elif expiration > safety_deadline:
        hours = (expiration - now_utc()).total_seconds() / 3600
        score += min(2.0, max(0.5, hours / 48.0))
        reasons.append(f"expires in {hours:.1f}h")

    if context >= min_context:
        score += 1.0
        reasons.append(f"context >= {min_context}")
    if context >= 128_000:
        score += 1.0
        reasons.append("large context >=128k")
    if context >= 1_000_000:
        score += 0.5
        reasons.append("very large context >=1M")

    kw_hits = sorted({kw for kw in CODING_KEYWORDS if kw in text})
    if kw_hits:
        score += min(2.5, 0.45 * len(kw_hits))
        reasons.append("coding keywords: " + ", ".join(kw_hits[:6]))

    neg_hits = sorted({kw for kw in NEGATIVE_KEYWORDS if kw in text})
    if neg_hits:
        score -= min(2.0, 0.6 * len(neg_hits))
        reasons.append("negative keywords: " + ", ".join(neg_hits[:4]))

    if "tools" in supported:
        score += 2.0
        reasons.append("supports tools")
    if "tool_choice" in supported:
        score += 0.5
        reasons.append("supports tool_choice")
    if "response_format" in supported or "structured_outputs" in supported:
        score += 1.0
        reasons.append("supports structured output")
    if "max_tokens" in supported:
        score += 0.25
    if "temperature" in supported:
        score += 0.25

    if "text" in input_modalities and "text" in output_modalities:
        score += 0.5
        reasons.append("text input/output")

    # Prefer known coding model families without hard requiring them.
    family_bonus = {
        "deepseek": 1.2,
        "qwen": 1.1,
        "coder": 1.2,
        "codestral": 1.2,
        "kimi": 0.7,
        "glm": 0.7,
        "gemma": 0.6,
        "mistral": 0.6,
        "nemotron": 0.5,
        "poolside": 1.0,
        "cobuddy": 1.0,
    }
    for key, bonus in family_bonus.items():
        if key in text:
            score += bonus
            reasons.append(f"family bonus: {key}")
            break

    return round(score, 3), reasons


def filter_and_rank(models: list[dict[str, Any]], min_context: int, safety_hours: int) -> list[dict[str, Any]]:
    deadline = now_utc() + dt.timedelta(hours=safety_hours)
    candidates: list[dict[str, Any]] = []
    for m in models:
        expiration = parse_expiration(m.get("expiration_date"))
        context = int(m.get("context_length") or 0)
        if not pricing_is_zero(m):
            continue
        if context < min_context:
            continue
        if expiration is not None and expiration <= deadline:
            continue
        score, reasons = score_model(m, deadline, min_context)
        candidates.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "canonical_slug": m.get("canonical_slug"),
            "context_length": context,
            "expiration_date": m.get("expiration_date"),
            "expiration_iso": expiration.isoformat() if expiration else None,
            "supported_parameters": m.get("supported_parameters") or [],
            "architecture": m.get("architecture") or {},
            "score": score,
            "score_reasons": reasons,
        })
    candidates.sort(key=lambda x: (x["score"], x["context_length"] or 0, str(x["id"])), reverse=True)
    return candidates


def benchmark_model(model_id: str, api_key: str, timeout: int = 45) -> dict[str, Any]:
    results = []
    total_score = 0.0
    total_latency = 0.0
    for item in BENCHMARK_PROMPTS:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a precise coding assistant. Follow output format exactly."},
                {"role": "user", "content": item["prompt"]},
            ],
            "max_tokens": 350,
            "temperature": 0.1,
        }
        start = time.time()
        try:
            data = fetch_json(OPENROUTER_CHAT_URL, api_key=api_key, payload=payload, timeout=timeout)
            latency = time.time() - start
            content = get_nested(data, "choices", [])
            text = ""
            if content and isinstance(content, list):
                text = ((content[0] or {}).get("message") or {}).get("content") or ""
            check_score = 0.0
            lower = text.lower()
            for check in item["checks"]:
                if check.lower() in lower:
                    check_score += 1.0
            if text.strip():
                check_score += 1.0
            if len(text) < 2500:
                check_score += 0.5
            if "```" in text and item["name"] == "json_exact":
                check_score -= 1.0
            results.append({
                "name": item["name"],
                "ok": True,
                "latency_seconds": round(latency, 3),
                "score": round(check_score, 3),
                "usage": data.get("usage"),
                "sample": text[:500],
            })
            total_score += check_score
            total_latency += latency
        except Exception as exc:  # noqa: BLE001 - report, don't crash whole selector
            latency = time.time() - start
            results.append({
                "name": item["name"],
                "ok": False,
                "latency_seconds": round(latency, 3),
                "score": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    avg_latency = total_latency / max(1, len([r for r in results if r.get("ok")]))
    return {
        "benchmark_score": round(total_score, 3),
        "avg_latency_seconds": round(avg_latency, 3),
        "results": results,
    }


def read_current_model(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    text = config_path.read_text()
    in_model = False
    for line in text.splitlines():
        if re.match(r"^model:\s*$", line):
            in_model = True
            continue
        if in_model and re.match(r"^[A-Za-z0-9_].*:\s*", line):
            in_model = False
        if in_model:
            m = re.match(r"^\s*default:\s*[\"']?([^\"'\n#]+)", line)
            if m:
                return m.group(1).strip()
    return None


def update_model_default(config_path: Path, model_id: str) -> None:
    text = config_path.read_text() if config_path.exists() else "model:\n"
    lines = text.splitlines()
    out = []
    in_model = False
    replaced = False
    inserted = False
    for i, line in enumerate(lines):
        if re.match(r"^model:\s*$", line):
            in_model = True
            out.append(line)
            continue
        if in_model and re.match(r"^[A-Za-z0-9_].*:\s*", line):
            if not replaced:
                out.append(f"  default: {model_id}")
                inserted = True
                replaced = True
            in_model = False
        if in_model and re.match(r"^\s*default:\s*", line):
            out.append(f"  default: {model_id}")
            replaced = True
            continue
        out.append(line)
    if in_model and not replaced:
        out.append(f"  default: {model_id}")
        inserted = True
    if not any(re.match(r"^model:\s*$", l) for l in lines):
        out.insert(0, f"model:\n  default: {model_id}")
    config_path.write_text("\n".join(out).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/opt/data/config.yaml")
    parser.add_argument("--report", default="/reports/model-selection.json")
    parser.add_argument("--mode", choices=["report_only", "apply"], default=os.getenv("SELECTOR_MODE", "report_only"))
    parser.add_argument("--safety-buffer-hours", type=int, default=int(os.getenv("SELECTOR_SAFETY_BUFFER_HOURS", "12")))
    parser.add_argument("--min-context", type=int, default=int(os.getenv("SELECTOR_MIN_CONTEXT", "32768")))
    parser.add_argument("--max-benchmark-models", type=int, default=int(os.getenv("SELECTOR_MAX_BENCHMARK_MODELS", "5")))
    parser.add_argument("--enable-benchmark", action="store_true", default=os.getenv("SELECTOR_ENABLE_BENCHMARK", "0").lower() in {"1", "true", "yes"})
    parser.add_argument("--switch-margin", type=float, default=float(os.getenv("SELECTOR_SWITCH_MARGIN", "1.5")))
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    config_path = Path(args.config)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    status = "ok"
    errors: list[str] = []
    try:
        models_data = fetch_json(OPENROUTER_MODELS_URL, api_key=api_key or None, timeout=30)
        models = models_data.get("data") or []
    except Exception as exc:
        status = "error"
        models = []
        errors.append(f"models_fetch_failed: {type(exc).__name__}: {exc}")

    current_model = read_current_model(config_path)
    candidates = filter_and_rank(models, min_context=args.min_context, safety_hours=args.safety_buffer_hours) if models else []

    # Optional live benchmark of the top static candidates. Disabled by default to avoid rate-limit/cost surprises.
    if args.enable_benchmark and api_key and candidates:
        for cand in candidates[: max(1, args.max_benchmark_models)]:
            bench = benchmark_model(str(cand["id"]), api_key=api_key)
            cand["benchmark"] = bench
            # Benchmark is intentionally strong but bounded; low latency earns a small bonus.
            latency_penalty = min(2.0, (bench.get("avg_latency_seconds") or 20) / 10)
            cand["score"] = round(float(cand["score"]) + float(bench["benchmark_score"]) - latency_penalty, 3)
            cand["score_reasons"].append(f"benchmark_score={bench['benchmark_score']} latency_penalty={latency_penalty:.2f}")
        candidates.sort(key=lambda x: (x["score"], x["context_length"] or 0, str(x["id"])), reverse=True)

    selected = candidates[0] if candidates else None
    current_candidate = next((c for c in candidates if c.get("id") == current_model), None)
    action = "none"
    should_switch = False

    if selected:
        if not current_model:
            should_switch = True
            action = "set_initial_model"
        elif current_model == selected["id"]:
            action = "keep_current_best"
        elif current_candidate is None:
            should_switch = True
            action = "switch_current_not_usable_free"
        elif float(selected["score"]) >= float(current_candidate["score"]) + args.switch_margin:
            should_switch = True
            action = "switch_better_score_margin"
        else:
            action = "keep_current_within_margin"
    else:
        status = "no_usable_free_model" if status == "ok" else status
        action = "keep_current_no_candidate"

    applied = False
    if args.mode == "apply" and selected and should_switch:
        update_model_default(config_path, str(selected["id"]))
        applied = True

    report = {
        "generated_at": now_utc().isoformat(),
        "status": status,
        "mode": args.mode,
        "errors": errors,
        "openrouter_models_seen": len(models),
        "free_candidates_seen": len(candidates),
        "current_model_before": current_model,
        "selected_model": selected.get("id") if selected else None,
        "selected": selected,
        "current_candidate": current_candidate,
        "action": action,
        "applied": applied,
        "config_path": str(config_path),
        "safety_buffer_hours": args.safety_buffer_hours,
        "min_context": args.min_context,
        "switch_margin": args.switch_margin,
        "benchmark_enabled": bool(args.enable_benchmark and api_key),
        "top_candidates": candidates[:20],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": report["status"],
        "mode": args.mode,
        "action": action,
        "applied": applied,
        "current_model_before": current_model,
        "selected_model": report["selected_model"],
        "free_candidates_seen": len(candidates),
        "report": str(report_path),
    }, ensure_ascii=False))
    return 0 if status in {"ok", "no_usable_free_model"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
