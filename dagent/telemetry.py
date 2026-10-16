#!/usr/bin/env python3
"""Timing and API usage summaries for DAGent runs."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time
import uuid

GEN_STAGES = ('understanding', 'draft', 'refinement', 'layout')


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def write_atomic(path, content):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.replace(path)


def export_csv(path, rows, fields):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(path, out.getvalue())


class Telemetry:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.current_paper = None

    def emit(self, record):
        record['recorded_utc'] = datetime.now(timezone.utc).isoformat()
        with (self.directory / 'timings.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def span(self, label, paper_id=None, **extra):
        event = dict(span_id=uuid.uuid4().hex, label=label, paper_id=paper_id,
                     started_at=time.time(), **extra)
        started = time.perf_counter()
        self.emit(dict(event, event='start'))
        outcome = 'returned'
        try:
            yield
        except BaseException as exc:
            outcome = type(exc).__name__
            raise
        finally:
            self.emit(dict(event, event='end', outcome=outcome,
                           finished_at=time.time(), wall_seconds=time.perf_counter()-started))


def summarize(run_dir, logs):
    protocol = read_json(run_dir / 'protocol.json')
    if protocol is None:
        return
    spans, ends = {}, []
    timing_path = logs / 'timings.jsonl'
    for line in timing_path.read_text().splitlines() if timing_path.exists() else []:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed process may leave a trailing partial record.
        if event.get('event') not in ('start', 'end'):
            continue
        if event['event'] == 'start':
            spans[event['span_id']] = event
        else:
            spans.pop(event['span_id'], None)
            ends.append(event)

    def duration(paper, label, fresh=False):
        selected = [e['wall_seconds'] for e in ends
                    if e.get('paper_id') == paper and e['label'] == label
                    and (not fresh or not e.get('reused', False))]
        return sum(selected) if selected else None

    counts = ('calls', 'provider_input_tokens', 'provider_output_tokens',
              'provider_total_tokens', 'unreported_usage_calls', 'reserved_tokens_unreported')
    totals = {scope: {k: 0 for k in counts} for scope in ('generation', 'evaluation')}
    calls, stages, papers = [], [], []
    for entry in protocol['entries']:
        pid = entry['paper_id']
        folder = run_dir / 'generation' / pid
        status = read_json(folder / 'status.json', {'status': 'pending'})
        pre = read_json(run_dir / 'input_preflight' / pid / 'status.json', {})
        paper = dict(paper_id=pid, status=status['status'], preflight_status=pre.get('status'),
                     failed_stage=status.get('failed_stage'),
                     failure_code=status.get('failure_code') or status.get('error_type'),
                     preflight_wall_seconds=duration(pid, 'preflight', fresh=True),
                     generation_wall_seconds=duration(pid, 'generation'),
                     evaluation_wall_seconds=duration(pid, 'evaluation'))
        for scope in totals:
            paper.update({scope + '_' + k: 0 for k in counts})
        for stage in (*GEN_STAGES, 'evaluation'):
            path = folder / ('evaluation/cost.json' if stage == 'evaluation'
                             else 'stages/' + stage + '/cost.json')
            if not path.is_file():
                continue
            ledger = read_json(path)
            scope = 'evaluation' if stage == 'evaluation' else 'generation'
            row = dict(paper_id=pid, stage=stage, paper_status=status['status'],
                       wall_seconds=duration(pid, stage),
                       ledger_runtime_seconds=ledger.get('runtime_seconds'),
                       api_elapsed_seconds=0.0, missing_api_latency_calls=0,
                       ledger_total_tokens=ledger.get('total_tokens'),
                       usage_is_exact=ledger.get('usage_is_exact'), ledger_path=str(path),
                       **{k: 0 for k in counts})
            for index, event in enumerate(ledger.get('events', []), 1):
                exact = event.get('usage_source') == 'provider'
                inp, out = event.get('input_tokens', 0), event.get('output_tokens', 0)
                row['calls'] += 1
                if exact:
                    row['provider_input_tokens'] += inp
                    row['provider_output_tokens'] += out
                    row['provider_total_tokens'] += inp + out
                else:
                    row['unreported_usage_calls'] += 1
                    row['reserved_tokens_unreported'] += inp + out
                elapsed = event.get('elapsed_seconds')
                row['api_elapsed_seconds'] += elapsed or 0
                row['missing_api_latency_calls'] += elapsed is None
                calls.append(dict(paper_id=pid, stage=stage, call_index=index,
                    model=event.get('model'), returned_model=event.get('returned_model'),
                    api_stage=event.get('stage'), status=event.get('status'),
                    usage_source=event.get('usage_source'), vision=event.get('vision'),
                    input_tokens_recorded=inp, output_tokens_recorded=out,
                    provider_total_tokens=(inp + out) if exact else None,
                    reserved_input_tokens=event.get('reserved_input_tokens'),
                    reserved_output_tokens=event.get('reserved_output_tokens'),
                    started_at=event.get('started_at'), elapsed_seconds=elapsed,
                    response_id=event.get('response_id'), request_hash=event.get('request_hash'),
                    provider_usage_json=json.dumps(event.get('provider_usage'), ensure_ascii=False),
                    ledger_path=str(path)))
            for key in counts:
                totals[scope][key] += row[key]
                paper[scope + '_' + key] += row[key]
            stages.append(row)
        papers.append(paper)
    for name, rows, fallback in (('api_calls.csv', calls, ['paper_id','stage','call_index']),
                                ('stage_metrics.csv', stages, ['paper_id','stage']),
                                ('paper_metrics.csv', papers, ['paper_id','status'])):
        export_csv(logs / name, rows, list(rows[0]) if rows else fallback)
    summary = dict(denominator=len(papers), full_benchmark=protocol.get('full_benchmark'),
        states={s: sum(p['status'] == s for p in papers)
                for s in sorted({p['status'] for p in papers})},
        totals=totals, incomplete_spans=list(spans.values()),
        command_wall_seconds={label: sum(e['wall_seconds'] for e in ends if e['label'] == label)
                              for label in sorted({e['label'] for e in ends if e['label'].startswith('command:')})},
        notes=['Provider totals exclude calls without reported usage; those reservations are separate.',
               'Nested wall times overlap: do not add command, paper, stage and render durations.',
               'Paper generation time excludes separately timed input preflight.',
               'Missing timers remain null; hard-killed spans remain incomplete.',
               'All cohort papers, including failed and pending papers, stay in the denominator.'])
    write_atomic(logs / 'metrics_summary.json', json.dumps(summary, ensure_ascii=False, indent=2))


