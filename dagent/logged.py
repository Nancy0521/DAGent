#!/usr/bin/env python3
"""Run DAGent with timing and token accounting. Paid generation is opt in."""
from __future__ import annotations

import argparse
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
from threading import Lock
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dagent import telemetry as base
from dagent.delivery import check_delivery

DEFAULT_CONFIG = ROOT / 'configs/dagent.json'


class Telemetry(base.Telemetry):
    """Preflight uses threads; serialize durable JSONL writes."""
    def __init__(self, directory):
        super().__init__(directory)
        self.lock = Lock()

    def emit(self, record):
        with self.lock:
            super().emit(record)


def instrument(module, telemetry):
    paper = ContextVar('dagent_timing_paper', default=None)
    originals = {}

    def install(name, fn):
        original = getattr(module, name)
        originals[name] = original
        setattr(module, name, wraps(original)(fn))

    original_generate = module.generate_one

    def generate(entry, *args, **kwargs):
        token = paper.set(entry['paper_id'])
        try:
            with telemetry.span('generation', entry['paper_id']):
                return original_generate(entry, *args, **kwargs)
        finally:
            paper.reset(token)

    original_preflight = module.ensure_preflight

    def preflight(run_dir, entry, *args, **kwargs):
        reused = (Path(run_dir) / 'input_preflight' / entry['paper_id']).exists()
        with telemetry.span('preflight', entry['paper_id'], reused=reused):
            return original_preflight(run_dir, entry, *args, **kwargs)

    install('generate_one', generate)
    install('ensure_preflight', preflight)

    def measured(fn, label):
        def call(*args, **kwargs):
            with telemetry.span(label, paper.get()):
                return fn(*args, **kwargs)
        return call

    # Wrap the audited understanding stage, not its individual retry calls.
    for name, label in (('select_evidence_with_audit', 'understanding'),
                        ('construct_provenance_storyboard', 'draft'), ('revise_grounded_storyboard', 'refinement'),
                        ('repair_poster', 'layout'), ('verify_repair_candidate', 'verify_repair_candidate')):
        install(name, measured(getattr(module, name), label))
    return originals


def snapshot_tools(logs):
    paths = [Path(__file__), Path(base.__file__), ROOT / 'dagent/delivery.py']
    directory = logs / 'logging_source_snapshot'
    directory.mkdir(exist_ok=True)
    hashes = {}
    for path in paths:
        content = path.read_bytes()
        snapshot = directory / path.name
        if snapshot.exists() and snapshot.read_bytes() != content:
            raise RuntimeError('Logging tools changed; use a new run to preserve provenance: ' + path.name)
        hashes[path.name] = hashlib.sha256(content).hexdigest()
    for path in paths:
        target = directory / path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())
    base.write_atomic(directory / 'sha256.json', json.dumps(hashes, indent=2))
    return hashes


def summarize(run_dir, logs):
    base.summarize(run_dir, logs)
    summary = base.read_json(logs / 'metrics_summary.json')
    if summary is None:
        return None
    workflow = []
    timing_path = logs / 'timings.jsonl'
    for line in timing_path.read_text().splitlines() if timing_path.exists() else []:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('event') == 'end' and event.get('label') == 'command:benchmark':
            workflow.append(event)
    summary['benchmark_invocations'] = workflow
    summary['benchmark_wall_seconds'] = sum(e['wall_seconds'] for e in workflow) if workflow else None
    summary['latest_benchmark_wall_seconds'] = workflow[-1]['wall_seconds'] if workflow else None
    # Only a fresh, fully measured invocation is a from-inputs wall-time result.
    summary['fresh_full_generation_wall_seconds'] = next((e['wall_seconds'] for e in reversed(workflow)
        if e.get('new_run') and e.get('paid_generation') and e.get('outcome') == 'returned'), None)
    summary['delivery'] = base.read_json(run_dir / 'delivery_check.json')
    summary['notes'] += [
        'benchmark_wall_seconds sums whole invocation spans, not nested stage or per-paper times.',
        'Workflow time includes prepare, preflight, generation, rendering, report and delivery checks; CSV export is outside the span.',
        'Resumed runs measure only work done in each invocation; old uninstrumented work is not reconstructed.',
        'All attempted candidates and recovery calls present in the native ledgers are counted, not just the selected poster.',
        'No paid quality evaluation is started by this launcher.',
        'Dollar amounts use configured model prices, not a provider invoice; missing prices or usage are not zero cost.',
    ]
    protocol = base.read_json(run_dir / 'protocol.json')
    for scope, totals in summary['totals'].items():
        stages = base.GEN_STAGES if scope == 'generation' else ('evaluation',)
        known_usd, missing_price_calls = 0.0, 0
        for entry in protocol['entries']:
            folder = run_dir / 'generation' / entry['paper_id']
            for stage in stages:
                path = folder / ('evaluation/cost.json' if stage == 'evaluation'
                                 else f'stages/{stage}/cost.json')
                for event in base.read_json(path, {}).get('events', []):
                    if event.get('usage_source') != 'provider':
                        continue
                    if event.get('usd') is None:
                        missing_price_calls += 1
                    else:
                        known_usd += event['usd']
        totals['usage_is_exact'] = totals['unreported_usage_calls'] == 0
        totals['known_usage_cost_usd'] = known_usd
        totals['missing_price_calls'] = missing_price_calls
        totals['total_cost_usd'] = (known_usd if totals['usage_is_exact'] and not missing_price_calls else None)
    base.write_atomic(logs / 'metrics_summary.json', json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def benchmark(args, telemetry):
    run_dir = Path(args.run_dir).resolve()
    module = importlib.import_module('dagent.agents.runner')
    originals = instrument(module, telemetry)

    def stage(name, *extra):
        print(f'\n=== {name} ===', flush=True)
        with telemetry.span('command:' + name):
            module.main([name, '--run-dir', str(run_dir), *extra])

    try:
        if not run_dir.exists():
            dataset_args = (('--manifest', args.manifest) if args.manifest else
                            ('--benchmark-root', args.benchmark_root))
            stage('prepare', *dataset_args, '--config', args.config)
        stage('preflight', '--workers', str(args.workers))
        if args.allow_paid_generation:
            stage('run', '--allow-paid-generation')
        stage('report')
        with telemetry.span('command:delivery_check'):
            delivery = check_delivery(run_dir, require_generation=args.allow_paid_generation)
            base.write_atomic(run_dir / 'delivery_check.json', json.dumps(delivery, ensure_ascii=False, indent=2))
            print(json.dumps(delivery, ensure_ascii=False, indent=2), flush=True)
            if not delivery['passed']:
                raise SystemExit(2)
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--allow-paid-generation', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--manifest', default=os.getenv('DAGENT_MANIFEST') or None,
                        help='Explicit manifest override; default: inventory benchmark/papers')
    parser.add_argument('--benchmark-root', default=os.getenv('DAGENT_BENCHMARK_ROOT', str(ROOT / 'benchmark')))
    parser.add_argument('--config', default=os.getenv('DAGENT_CONFIG', str(DEFAULT_CONFIG)))
    parser.add_argument('--workers', type=int, default=int(os.getenv('DAGENT_PREFLIGHT_WORKERS', '2')))
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    logs = run_dir.with_name(run_dir.name + '_logs')
    telemetry = Telemetry(logs)
    # Only the metadata below contains CLI parameters, never credentials or env values.
    hashes = snapshot_tools(logs)
    telemetry.emit(dict(event='metadata', span_id=uuid.uuid4().hex, tool_hashes=hashes,
                        argv=list(sys.argv[1:] if argv is None else argv),
                        python=sys.version, executable=sys.executable, pid=os.getpid()))
    try:
        if not args.summarize_only:
            with telemetry.span('command:benchmark', new_run=not run_dir.exists(),
                                paid_generation=args.allow_paid_generation):
                benchmark(args, telemetry)
    finally:
        already_failing = sys.exc_info()[0] is not None
        try:
            summary = summarize(run_dir, logs)
            if summary is not None:
                print('\n=== 总统计（评测另计） ===', flush=True)
                print(json.dumps({key: summary[key] for key in ('denominator', 'states', 'totals',
                    'benchmark_wall_seconds', 'latest_benchmark_wall_seconds', 'command_wall_seconds')},
                    ensure_ascii=False, indent=2), flush=True)
            print(f'\n统计目录：{logs}', flush=True)
        except Exception as exc:
            print('Telemetry export failed:', type(exc).__name__, str(exc), file=sys.stderr)
            if not already_failing:
                raise


if __name__ == '__main__':
    # Persist timers/ledgers on normal termination; SIGKILL cannot run cleanup.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
