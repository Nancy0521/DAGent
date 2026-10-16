"""Usage accounting around unchanged reference chat-completion requests."""
from __future__ import annotations
import hashlib
import json
import os
from types import SimpleNamespace
from dagent.core.api import APIError, safe_error_metadata
from dagent.core.config import dump_json


class ReferenceClient:
    def __init__(self, config, ledger, logs, transport=None):
        self.config, self.ledger, self.logs, self.transport = config, ledger, logs, transport
        self.stage = "eval_reference"
        self.client = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        model = self.config.vision_model
        if self.transport is None and self.client is None:
            if not os.getenv(model.api_key_env):
                raise APIError(f"Missing credential: {model.api_key_env}")
            from openai import OpenAI
            # Retries are owned by the copied reference functions and each call is billed.
            self.client = OpenAI(api_key=os.environ[model.api_key_env],
                base_url=os.getenv(model.base_url_env) or None, timeout=model.timeout_seconds, max_retries=0)
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        text, images = [], 0
        for message in request["messages"]:
            content = message["content"]
            if isinstance(content, str):
                text.append(content)
            else:
                text.extend(p["text"] for p in content if p["type"] == "text")
                images += sum(p["type"] == "image_url" for p in content)
        reserve = len("".join(text).encode()) + 128 + images*self.config.budget.image_token_reservation
        event = self.ledger.reserve(self.stage, model, reserve, request["max_tokens"], bool(images), digest)
        path = self.logs / f"{len(self.ledger.events):03d}_{self.stage}.json"
        # Do not duplicate image data URLs in the log.
        log = dict(stage=self.stage, request_hash=digest, model=request["model"], prompt=text,
                   images=images, max_tokens=request["max_tokens"], status="in_flight",
                   request_options={k:v for k,v in request.items() if k not in ("model", "messages")})
        dump_json(path, log)
        response = None
        try:
            if self.transport is not None:
                raw = self.transport(request)
                if isinstance(raw, dict):
                    response = raw
                    msg = SimpleNamespace(**raw["choices"][0]["message"])
                    obj = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
                else:
                    obj = raw
                    response = obj.model_dump()
            else:
                obj = self.client.chat.completions.create(**request)
                response = obj.model_dump()
            self.ledger.settle(event, model, response.get("usage"), response_id=response.get("id"),
                               returned_model=response.get("model"))
            dump_json(path, dict(log, status="received", response=response))
            return obj
        except Exception as exc:
            self.ledger.settle(event, model, response.get("usage") if response else None,
                               status="error", error_type=type(exc).__name__, **safe_error_metadata(exc))
            dump_json(path, dict(log, status="error", error_type=type(exc).__name__, **safe_error_metadata(exc)))
            # Preserve exception type for the reference retry/parser behavior.
            raise
