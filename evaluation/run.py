"""Evaluate existing DAGent posters independently of generation code snapshots."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from dagent.core.config import ROOT, Config, dump_json, writable_path
from dagent.core.evaluation.benchmark_eval import evaluate
from dagent.core.evaluation.records import normalize_metrics, evaluation_failed


def summarize(rows):
    """Keep available Judge ratings, as in the comparison's fill-aspects report.

    An incomplete Judge still has usable ratings and must not discard an
    independently completed PaperQuiz. Failed/interrupted attempts and failed
    PaperQuiz buckets do not contribute scores.
    """
    columns = {k: [] for k in ('element', 'layout', 'engagement', 'clarity', 'content', 'logic',
        'aesthetic_avg', 'information_avg', 'overall_judge', 'verbatim_raw', 'interpretive_raw',
        'overall_raw', 'verbatim_aug', 'interpretive_aug', 'overall_aug')}
    def add(key, value):
        if type(value) in (int, float) and math.isfinite(value):
            columns[key].append(value)
    for row in rows:
        result = row.get('result') or {}
        if not result or result.get('status', 'complete') != 'complete':
            continue
        metrics = result.get('metrics') or {}
        judge = metrics.get('judge', {})
        for key, rating in judge.get('aspects', {}).items():
            if key in columns:
                add(key, rating.get('score'))
        for key in ('aesthetic_avg', 'information_avg'):
            add(key, judge.get(key))
        add('overall_judge', judge.get('overall'))
        quiz = metrics.get('paperquiz', {})
        if quiz.get('status', 'ok') != 'ok':
            continue
        for key in ('verbatim_raw', 'interpretive_raw', 'overall_raw', 'verbatim_aug', 'interpretive_aug', 'overall_aug'):
            add(key, quiz.get(key))
    return dict(total=len(rows), ok=sum(r.get('status') == 'ok' for r in rows),
                failed=sum(r.get('status') == 'fail' for r in rows),
                skipped=sum(r.get('status') == 'skipped' for r in rows),
                means={k: sum(v)/len(v) if v else None for k,v in columns.items()},
                valid_counts={k: len(v) for k,v in columns.items()},
                aggregation='Mean of available per-poster values from completed attempts. '
                            'Partial Judge ratings and completed PaperQuiz are counted independently; '
                            'failed/skipped PaperQuiz is excluded. Inspect valid_counts.')


def run(run_dir, manifest, config, metrics, *, report_only=False, retry_incomplete=False, transport=None):
    run_dir = writable_path(run_dir)
    entries = json.loads(Path(manifest).read_text())['papers']
    if not entries or len({e['paper_id'] for e in entries}) != len(entries):
        raise ValueError('Manifest must contain unique, nonempty paper IDs')
    metrics = normalize_metrics(metrics)
    rows = []
    for entry in entries:
        paper_id = entry['paper_id']
        if Path(paper_id).name != paper_id or paper_id in ('.', '..'):
            raise ValueError('Unsafe paper ID')
        folder = writable_path(run_dir/'generation'/paper_id)
        row = dict(paper_id=paper_id, status='skipped')
        path = folder/'evaluation'/'metrics.json'
        if report_only:
            if path.is_file():
                result = json.loads(path.read_text())
                row.update(status='fail' if evaluation_failed(result) else 'ok', result=result)
            else:
                row['reason'] = 'evaluation_report_missing'
        elif not (folder/'poster.png').is_file():
            row['reason'] = 'generated_poster_missing'
        else:
            try:
                result = evaluate(folder, entry, config, metrics, transport,
                                  retry_incomplete=retry_incomplete)
                row.update(status='fail' if evaluation_failed(result) else 'ok', result=result)
                dump_json(folder/'evaluation_result.json', result)
            except Exception as exc:
                row.update(status='fail', error_type=type(exc).__name__)
        rows.append(row)
    summary = summarize(rows)
    dump_json(run_dir/'evaluation_summary.json', summary)
    dump_json(run_dir/'evaluation_results.json', rows)
    name = {('qa',): 'paperquiz', ('judge',): 'llm_as_judge'}.get(metrics)
    if name is not None:
        dump_json(run_dir/f'{name}_summary.json', summary)
        dump_json(run_dir/f'{name}_results.json', rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', default='configs/dagent.json')
    parser.add_argument('--metrics', nargs='+', choices=['judge', 'qa'], default=['judge', 'qa'])
    parser.add_argument('--allow-paid-evaluation', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true', help='Explicitly allow another attempt for an incomplete result')
    args = parser.parse_args(argv)
    if not args.report_only and any(m in ('judge', 'qa') for m in args.metrics) and not args.allow_paid_evaluation:
        parser.error('Judge/PaperQuiz require --allow-paid-evaluation')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    summary = run(args.run_dir, args.manifest, Config.load(args.config), args.metrics,
                  report_only=args.report_only, retry_incomplete=args.retry_incomplete)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
