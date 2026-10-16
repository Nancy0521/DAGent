"""Judge/PaperQuiz core restored from the comparison workflow's Python 3.10 cache.

Request construction and parsing retain the cached behavior, including its escaped
Judge regex. The detailed fill stage in judge_fill.py recovers missing ratings.
See reference_sources.json for the source revision.
"""
from __future__ import annotations
import base64
import io
import json
import re
import time
from pathlib import Path
from PIL import Image
from evaluation.aspects import ASPECTS


def b64_jpeg(path, max_side=1280):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def merge_qa_bucket(bucket):
    questions = bucket.get("questions") or {}
    answers = bucket.get("answers") or {}
    out = {}
    for qid, q in questions.items():
        qq = dict(q) if isinstance(q, dict) else {"question": str(q)}
        ans = answers.get(qid, "")
        if isinstance(ans, str) and ans.strip():
            m = re.search("[ABCD]", ans.strip().upper())
            qq["answer"] = m.group(0) if m else ans.strip()[:1].upper()
        out[qid] = qq
    return out


def chat_json(client, model, b64, prompt, max_tokens):
    if "deepseek" in (model or "").lower() and max_tokens < 512:
        max_tokens = 1024
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return content
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    if not reasoning:
        dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
        reasoning = str(dump.get("reasoning_content") or "").strip()
    if not reasoning:
        return ""
    m = re.search(r"\{[\s\S]*\}", reasoning)
    if m:
        return m.group(0)
    m = re.search(r"\b([ABCD])\b", reasoning.upper())
    if m:
        return m.group(1)
    return reasoning


def judge_six(client, model, b64):
    aspects = {}
    for key, name in ASPECTS:
        prompt = (
            f"You are a scientific poster judge. Score ONLY '{name}' from 1 to 5 "
            f"(integers or .5 allowed). "
            f'Return ONLY JSON: {{"score": <number>, "reason": "<one short sentence>"}}'
        )
        raw, score, reason = "", None, ""
        for attempt in range(3):
            try:
                raw = chat_json(client, model, b64, prompt, max_tokens=1024)
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    obj = json.loads(m.group(0))
                    score = float(obj.get("score"))
                    reason = str(obj.get("reason") or "")[:200]
                break
            except Exception as e:
                reason = f"err:{type(e).__name__}:{str(e)[:120]}"
                time.sleep(1.5 * (attempt + 1))
        aspects[key] = {"score": score, "reason": reason, "raw": raw[:300]}
        time.sleep(0.25)
    aes = [aspects[k]["score"] for k in ("element", "layout", "engagement") if aspects[k]["score"] is not None]
    info = [aspects[k]["score"] for k in ("clarity", "content", "logic") if aspects[k]["score"] is not None]
    alls = [v["score"] for v in aspects.values() if v["score"] is not None]
    return {"aspects": aspects, "aesthetic_avg": sum(aes)/len(aes) if aes else None,
            "information_avg": sum(info)/len(info) if info else None,
            "overall": sum(alls)/len(alls) if alls else None}


def run_quiz_bucket(client, model, b64, questions, limit):
    items = list(questions.items())
    if limit is not None:
        items = items[:limit]
    if not items:
        return float("nan"), 0, 0
    correct = asked = 0
    for qid, q in items:
        stem = q.get("question") or q.get("Question") or ""
        opts = q.get("options") or []
        ans = str(q.get("answer") or "").strip().upper()[:1]
        opt_txt = "\n".join(opts) if isinstance(opts, list) else str(opts)
        prompt = ("Answer the multiple-choice question using ONLY the poster image. "
                  f"Reply with a single letter A/B/C/D.\n\nQ: {stem}\n{opt_txt}")
        asked += 1
        pred = ""
        for attempt in range(3):
            try:
                raw = chat_json(client, model, b64, prompt, max_tokens=256)
                m = re.search(r"[ABCD]", raw.upper())
                pred = m.group(0) if m else ""
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if pred and ans and pred == ans:
            correct += 1
        time.sleep(0.2)
    return 100.0 * correct / max(1, asked), correct, asked


def density_aug(sr, L, W=774.0):
    if L <= 0:
        L = 1.0
    if W <= 0:
        W = 1.0
    return float(sr) * (1.0 + 1.0 / max(1.0, float(L) / float(W)))


def load_qa(path):
    """Select a matched per-paper nested QA blob, retaining separate answer keys."""
    qa = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(qa, dict) and "qa" in qa:
        qa = qa["qa"]
    if isinstance(qa, str):
        qa = json.loads(qa)
    if not isinstance(qa, dict) or not ({"detail", "understanding", "interpretive"} & set(qa)):
        raise ValueError("Expected a per-paper nested detail/understanding QA object")
    buckets = []
    for bucket in (qa.get("detail", {}), qa.get("understanding", qa.get("interpretive", {}))):
        if isinstance(bucket, str):
            bucket = json.loads(bucket)
        buckets.append(merge_qa_bucket(bucket))
    return buckets


def paperquiz(client, model, b64, buckets, limit=50, text_len_words_proxy=400.0):
    """Comparison main's calculation, with explicit per-paper input association."""
    v, vc, vn = run_quiz_bucket(client, model, b64, buckets[0], limit)
    i, ic, inn = run_quiz_bucket(client, model, b64, buckets[1], limit)
    va = density_aug(v, text_len_words_proxy)
    ia = density_aug(i, text_len_words_proxy)
    result = dict(verbatim_raw=v, interpretive_raw=i, overall_raw=(v+i)/2,
                  verbatim_aug=va, interpretive_aug=ia, overall_aug=(va+ia)/2,
                  verbatim_correct=vc, verbatim_n=vn, interpretive_correct=ic, interpretive_n=inn,
                  text_len_words_proxy=text_len_words_proxy, reader_model=model)
    # Preserve the reference failure policy while retaining diagnostic scores.
    result["status"] = "incomplete" if vc+ic == 0 and vn+inn > 0 else "ok"
    if result["status"] == "incomplete":
        result["reason"] = "quiz all wrong/empty; likely API fail (reference policy)"
    return result
