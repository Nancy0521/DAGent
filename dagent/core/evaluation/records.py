"""Local evaluation provenance, immutable attempts and no-surprise resume behavior."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os

from dagent.core.config import dump_json, writable_path
from dagent.core.document import sha256

METRICS = ('judge', 'qa')


def normalize_metrics(metrics):
    if isinstance(metrics, str):
        raise ValueError('Supply a sequence of evaluation metrics, not a string')
    requested = set(metrics)
    if not requested or requested - set(METRICS):
        raise ValueError('Select one or more metrics from judge, qa')
    return tuple(name for name in METRICS if name in requested)


def file_identity(path):
    if not path:
        return None
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path) if path.is_file() else None)


def input_identity(run_dir, entry, config, metrics, aspects):
    """Hash metric inputs, not only their paths; never persist API credentials."""
    sources = dict(poster=file_identity(run_dir/'poster.png'))
    from dagent.core.config import ROOT
    reference = ROOT/'evaluation'
    sources['implementation'] = {str(p.relative_to(ROOT)): file_identity(p)
        for p in sorted([*reference.glob('*.py'), reference/'reference_sources.json',
                         *Path(__file__).parent.glob('*.py')])}
    if 'judge' in metrics:
        sources['judge_prompts'] = {name: file_identity(reference/'judge_prompts'/(name+'.yaml')) for name in aspects}
    if 'qa' in metrics:
        sources['questions'] = file_identity(entry.get('qa_path'))
        sources['ocr_cache'] = file_identity(entry.get('ocr_cache_path'))
    if 'judge' in metrics or 'qa' in metrics:
        endpoint = os.getenv(config.vision_model.base_url_env, '')
        sources['provider_identity_sha256'] = hashlib.sha256(endpoint.encode()).hexdigest()
    return sources


@contextmanager
def evaluation_lock(canonical):
    """Process-scoped POSIX lock; a crashed evaluator cannot leave a stale lock."""
    import fcntl
    canonical = writable_path(canonical)
    canonical.mkdir(parents=True, exist_ok=True)
    with writable_path(canonical/'.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another evaluation is active for this poster') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def aggregate_cost(canonical, *, save=True):
    """Rebuild from every attempt ledger, even failed/interrupted reservations."""
    paths = sorted((canonical/'attempts').glob('*/cost.json'))
    attempts = [json.loads(path.read_text()) for path in paths]
    result = dict(attempts=len(attempts), generation_budget_excludes_this=True,
                  total_tokens=sum(a['total_tokens'] for a in attempts),
                  calls=sum(a['calls'] for a in attempts),
                  usd=sum(a['usd'] for a in attempts) if all(a.get('usd') is not None for a in attempts) else None,
                  usage_is_exact=all(a.get('usage_is_exact', False) for a in attempts),
                  attempt_cost_files=[str(path) for path in paths])
    if save:
        dump_json(canonical/'cost.json', result)
    return result


def cached_report(canonical, fingerprint, retry_incomplete=False):
    """Reuse older metric-set attempts too; never implicitly retry paid failures."""
    paths = sorted((canonical/'attempts').glob('*/metrics.json'), reverse=True)
    # Compatibility for reports without an archived attempt; current records live
    # in attempts first, and metrics.json is merely a selected-report pointer.
    if (canonical/'metrics.json').is_file():
        paths.append(canonical/'metrics.json')
    for path in paths:
        report = json.loads(path.read_text())
        if report.get('fingerprint') != fingerprint:
            continue
        if report.get('status') == 'running':
            # The caller holds the lock: an old running checkpoint is interrupted.
            report.update(status='interrupted', error_type='InterruptedAttempt')
            dump_json(path, report)
        def incomplete_metric(metric):
            if 'status' in metric:
                return metric['status'] != 'ok'
            return any(incomplete_metric(value) for value in metric.values() if isinstance(value, dict))
        incomplete = (report.get('status', 'complete') != 'complete'
                      or any(incomplete_metric(metric) for metric in report['metrics'].values()))
        if retry_incomplete and incomplete:
            return None
        dump_json(canonical/'metrics.json', report)
        return report
    return None


def evaluation_failed(report):
    """Optional missing data may be skipped; failed paid evaluations are errors."""
    return (report.get('status', 'complete') != 'complete'
            or any(metric.get('status') == 'incomplete' for metric in report['metrics'].values()))
