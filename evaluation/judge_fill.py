"""Detailed Judge fill functions copied from the baseline comparison workflow.

Only the prompt directory is relocated; request and scoring functions are retained.
"""
from __future__ import annotations
import base64
import json
import re
import time
from pathlib import Path
from typing import Any
import yaml
from openai import OpenAI
from evaluation.aspects import ASPECTS

PROMPT_DIR = Path(__file__).resolve().parent / "judge_prompts"
AES_KEYS = ["element", "layout", "engagement"]
INFO_KEYS = ["clarity", "content", "logic"]


def load_prompt(aspect_key: str) -> str:
    ypath = PROMPT_DIR / f"{aspect_key}.yaml"
    if not ypath.is_file():
        raise FileNotFoundError(ypath)
    cfg = yaml.safe_load(ypath.read_text())
    sys_p = (cfg.get("system_prompt") or "").strip()
    tmpl = (cfg.get("template") or "").strip()
    # Force machine-readable JSON (P2P example uses int; we allow .5)
    footer = (
        '\n\nReturn ONLY a JSON object (no markdown fences), exactly:\n'
        '{"score": <number 1-5, integers or .5 allowed>, '
        '"reason": "<one short sentence>"}\n'
        "Do not answer with a single letter. Do not include other keys."
    )
    return f"{sys_p}\n\n{tmpl}{footer}"


def png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    t = text.strip()
    # strip fences
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def looks_like_letter_only(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if re.fullmatch(r"[A-Da-d]", t):
        return True
    if re.fullmatch(r"[A-Da-d][.\s]*", t):
        return True
    return False


def chat_judge_json(
    client: OpenAI,
    model: str,
    b64: str,
    prompt: str,
    *,
    # Reasoning models spend tokens in thinking/reasoning_content; 1024 starves
    # the final JSON (finish=length, empty content). 2560 verified: emits JSON
    # with finish=stop on all remaining cases (2048 still truncated one case).
    max_tokens: int = 2560,
    use_json_mode: bool = True,
) -> str:
    """Judge-only chat: NEVER fall back to ABCD letters."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        # Some gateways reject response_format; retry without.
        if use_json_mode:
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return content
    # DeepSeek sometimes puts text only in reasoning_content — take JSON from there,
    # but never accept a bare A/B/C/D as a judge score.
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if not reasoning and hasattr(msg, "model_dump"):
        reasoning = str(msg.model_dump().get("reasoning_content") or "")
    reasoning = reasoning.strip()
    if not reasoning:
        return ""
    m = re.search(r"\{[\s\S]*\}", reasoning)
    if m:
        return m.group(0)
    return ""


def score_one_aspect(
    client: OpenAI,
    model: str,
    b64: str,
    aspect_key: str,
    *,
    max_attempts: int = 5,
) -> dict[str, Any]:
    prompt = load_prompt(aspect_key)
    raw = ""
    reason = ""
    score = None
    for attempt in range(max_attempts):
        try:
            raw = chat_judge_json(client, model, b64, prompt)
            if looks_like_letter_only(raw):
                reason = f"err:letter_only:{raw[:20]!r}"
                time.sleep(1.2 * (attempt + 1))
                continue
            obj = extract_json_obj(raw)
            if not obj:
                reason = "err:no_json"
                time.sleep(1.2 * (attempt + 1))
                continue
            sc = obj.get("score")
            score = float(sc)
            if not (1.0 <= score <= 5.0):
                reason = f"err:score_out_of_range:{score}"
                score = None
                time.sleep(1.2 * (attempt + 1))
                continue
            reason = str(obj.get("reason") or "")[:200]
            break
        except Exception as e:
            reason = f"err:{type(e).__name__}:{str(e)[:120]}"
            score = None
            time.sleep(1.5 * (attempt + 1))
    return {"score": score, "reason": reason, "raw": (raw or "")[:300]}


def recompute_avgs(aspects: dict[str, Any]) -> dict[str, Any]:
    def vals(keys: list[str]) -> list[float]:
        out = []
        for k in keys:
            v = aspects.get(k) or {}
            sc = v.get("score") if isinstance(v, dict) else None
            if isinstance(sc, (int, float)):
                out.append(float(sc))
        return out

    aes = vals(AES_KEYS)
    info = vals(INFO_KEYS)
    alls = vals([k for k, _ in ASPECTS])
    return {
        "aspects": aspects,
        "aesthetic_avg": (sum(aes) / len(aes)) if aes else None,
        "information_avg": (sum(info) / len(info)) if info else None,
        "overall": (sum(alls) / len(alls)) if alls else None,
    }


def missing_aspects(judge: dict[str, Any] | None) -> list[str]:
    if not judge:
        return [k for k, _ in ASPECTS]
    aspects = judge.get("aspects") or {}
    miss = []
    for k, _ in ASPECTS:
        v = aspects.get(k) or {}
        sc = v.get("score") if isinstance(v, dict) else None
        if not isinstance(sc, (int, float)):
            miss.append(k)
    return miss
