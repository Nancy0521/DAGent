"""Usage accounting with optional, explicitly configured admission limits.

Uncertain network failures are charged their reservation, never treated as free.
By default there is no generation budget cap. Explicit finite caps can be
configured for generation or independent evaluation.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from dagent.core.config import BudgetConfig, ModelConfig, dump_json


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, limits: BudgetConfig, path: Path | None = None):
        self.limits, self.path = limits, path
        self.events: list[dict] = []
        self.started = time.monotonic()
        self.renderer_calls = 0
        self.cache_hits = 0

    @staticmethod
    def dollars(model: ModelConfig, input_tokens: int, output_tokens: int):
        if None in (model.input_usd_per_million, model.output_usd_per_million):
            return None
        return (input_tokens*model.input_usd_per_million + output_tokens*model.output_usd_per_million)/1e6

    def totals(self, events=None):
        events = self.events if events is None else events
        ins = sum(e['input_tokens'] for e in events)
        outs = sum(e['output_tokens'] for e in events)
        known = all(e['usd'] is not None for e in events)
        return dict(input_tokens=ins, output_tokens=outs, total_tokens=ins+outs,
                    llm_calls=sum(not e['vision'] for e in events),
                    vlm_calls=sum(e['vision'] for e in events), calls=len(events),
                    usd=sum(e['usd'] for e in events) if known else None,
                    usage_is_exact=all(e['usage_source'] == 'provider' for e in events))

    def admit(self, stage: str, model: ModelConfig, input_tokens: int, output_tokens: int, vision=False):
        totals, b = self.totals(), self.limits
        projected = input_tokens + output_tokens
        if b.max_tokens is not None and totals['total_tokens'] + projected > b.max_tokens:
            raise BudgetExceeded(f"{stage}: total token reservation exceeds {b.max_tokens}")
        if ((b.max_calls is not None and totals['calls'] >= b.max_calls) or
                (vision and b.max_vlm_calls is not None and totals['vlm_calls'] >= b.max_vlm_calls)):
            raise BudgetExceeded(f"{stage}: API call budget exhausted")
        if b.max_tokens is not None and stage.startswith('content'):
            spent = self.totals([e for e in self.events if e['stage'].startswith('content')])['total_tokens']
            if spent + projected > b.max_tokens*b.content_fraction:
                raise BudgetExceeded(f"{stage}: content allocation exhausted; preserving layout budget")
        usd = self.dollars(model, input_tokens, output_tokens)
        if b.max_usd is not None and (usd is None or totals['usd'] is None or totals['usd']+usd > b.max_usd):
            raise BudgetExceeded(f"{stage}: dollar reservation exceeds budget")

    def reserve(self, stage, model, input_tokens, output_tokens, vision, request_hash):
        self.admit(stage, model, input_tokens, output_tokens, vision)
        event = dict(stage=stage, model=model.name, vision=vision, request_hash=request_hash,
                     input_tokens=input_tokens, output_tokens=output_tokens,
                     reserved_input_tokens=input_tokens, reserved_output_tokens=output_tokens,
                     usd=self.dollars(model, input_tokens, output_tokens), usage_source='reservation',
                     status='in_flight', billed_here=True, started_at=time.time())
        self.events.append(event)
        self.save()
        return event

    def settle(self, event, model, usage, status='ok', **metadata):
        if usage is not None:
            ins, outs = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if isinstance(ins, int) and isinstance(outs, int) and min(ins, outs) >= 0:
                event.update(input_tokens=ins, output_tokens=outs, usage_source='provider')
        event.update(status=status, elapsed_seconds=time.time()-event['started_at'], **metadata)
        event['usd'] = self.dollars(model, event['input_tokens'], event['output_tokens'])
        self.save()

    def snapshot(self):
        totals = self.totals()
        stages = {s: self.totals([e for e in self.events if e['stage'] == s]) for s in sorted({e['stage'] for e in self.events})}
        return dict(**totals, stages=stages,
                    billed_here=self.totals([e for e in self.events if e.get('billed_here', True)]),
                    shared_preparation_attribution=self.totals([e for e in self.events if not e.get('billed_here', True)]),
                    renderer_calls=self.renderer_calls,
                    critic_calls=sum(e['stage'].startswith('critic') for e in self.events),
                    cache_hits=self.cache_hits, runtime_seconds=time.monotonic()-self.started,
                    limits=asdict(self.limits), events=self.events,
                    budget_enforced=any(v is not None for v in (self.limits.max_tokens, self.limits.max_calls,
                                                               self.limits.max_vlm_calls, self.limits.max_usd)),
                    budget_overrun=((self.limits.max_tokens is not None and totals['total_tokens'] > self.limits.max_tokens) or
                        (self.limits.max_usd is not None and totals['usd'] is not None and totals['usd'] > self.limits.max_usd)),
                    cost_note='USD uses configured rates, not a provider invoice; unknown prices are null. '
                              'Unreported usage / failed requests retain conservative reservations.')

    def save(self):
        if self.path:
            dump_json(self.path, self.snapshot())
