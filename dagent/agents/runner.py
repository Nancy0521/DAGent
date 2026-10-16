"""DAGent: grounded poster generation with bounded render feedback."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dagent.core.api import APIClient, safe_error_metadata
from dagent.agents.dataset import DEFAULT_BENCHMARK_ROOT, discover_sources
from dagent.agents.writing_agent import construct_provenance_storyboard, validate_provenance_storyboard
from dagent.agents.grounded_revision import revise_grounded_storyboard
from dagent.agents.evidence_agent import (
    INPUT_POLICY, InputFailure, load_preflight, prepare_evidence_input, write_input_failure, visual_mode,
)
from dagent.core.benchmark import select_papers, validate_entry
from dagent.core.budget import BudgetLedger
from dagent.core.config import Config, dump_json, writable_path
from dagent.core.document import sha256, select_evidence_pointers
from dagent.core.provenance_binding import reroute_visual_bindings
from dagent.agents.recovery import resolve_pdf_input
from dagent.agents.layout import audit_layout, make_layout
from dagent.agents.renderer import PosterRenderer
from dagent.agents.repair_agent import SYSTEM, SCHEMA, apply_repair_decision, repair_context, validate_repair_decision
from dagent.agents.selection import quality_score, VERSION as SELECTION_POLICY
from dagent.agents.retry import ATTEMPT_LIMIT, FeedbackAPI, failure_feedback

VERSION = 'dagent'
CANDIDATE_LIMIT = ATTEMPT_LIMIT
PREFLIGHT_TIMEOUT_SECONDS = 180
DEFAULT_MANIFEST = None  # Build a fresh inventory of benchmark/papers for every new run.
DEFAULT_CONFIG = ROOT / 'configs/dagent.json'
PRODUCTS = ('poster.png', 'poster.pdf', 'poster.pptx', 'layout.json',
            'layout_search.json', 'geometry_check.json', 'render_checks.json', 'decision.json',
            'storyboard.json', 'evidence_route.json', 'quality_score.json')


class RenderedCandidateError(ValueError):
    """A candidate rendered to real files but failed post-render inspection."""

    def __init__(self, defects, search=None):
        self.defects = defects
        self.search = search
        super().__init__('Rendered text failed glyph inspection')


class LayoutGeometryError(ValueError):
    """A candidate failed the pre-render layout geometry gate."""

    def __init__(self, geometry, search=None):
        self.geometry = geometry
        self.defects = geometry.get('defects', [])
        self.search = search
        super().__init__('Layout geometry failed')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def run_lock(directory):
    with (directory / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This benchmark run is already active') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def sources():
    return sorted([*(ROOT / 'dagent').rglob('*.py'), *(ROOT / 'evaluation').rglob('*.py'),
                   *(ROOT / 'evaluation' / 'judge_prompts').glob('*.yaml')])


def referenced_assets(parsed_json):
    if not parsed_json:
        return {}
    path = Path(parsed_json)
    if not path.is_file():
        return {}
    try:
        records = read(path)
    except (ValueError, OSError):
        return {}  # Keep malformed inputs in the cohort; ingestion will report failure.
    assets = {}
    for record in records:
        if not isinstance(record, dict) or record.get('type') not in ('image', 'table'):
            continue
        if not isinstance(record.get('img_path'), str) or not record['img_path']:
            continue
        asset = (path.parent / record['img_path']).resolve()
        if asset.is_relative_to(path.parent.resolve()):
            assets[str(asset)] = sha256(asset) if asset.is_file() else None
    return assets


def file_hash(path):
    return sha256(Path(path)) if path and Path(path).is_file() else None


def check_input(entry):
    validate_entry(entry)
    parsed = entry.get('parsed_json')
    if parsed and file_hash(parsed) != entry['parsed_json_sha256']:
        raise ValueError('Parsed input changed or missing')
    # Cached images are not generation inputs. Their existence/decodability is not a gate.
    recovery = entry.get('input_recovery', {})
    original = recovery.get('original', {})
    if recovery.get('status') == 'recovered' and original.get('sha256') is not None:
        if file_hash(original['path']) != original['sha256']:
            raise InputFailure('original_input_changed', 'Original recovery input changed', 'input')


def check_evaluation_input(entry):
    for field in ('qa_path', 'ocr_cache_path'):
        if file_hash(entry.get(field)) != entry.get(field + '_sha256'):
            raise ValueError(f'Benchmark evaluation input changed: {field}')


def prepare(run_dir, manifest_path=DEFAULT_MANIFEST, config_path=DEFAULT_CONFIG,
            *, papers=None, limit=0, ready_only=False, benchmark_root=DEFAULT_BENCHMARK_ROOT):
    run_dir = writable_path(run_dir)
    manifest_path = Path(manifest_path).resolve() if manifest_path else None
    config_path = Path(config_path).resolve()
    manifest = read(manifest_path) if manifest_path else discover_sources(benchmark_root)
    selected, selection = select_papers(manifest, papers=papers, limit=limit, ready_only=ready_only)
    if not selected or len({e['paper_id'] for e in selected}) != len(selected):
        raise ValueError('Selection must contain unique, nonempty paper IDs')
    for entry in selected:
        if Path(entry['paper_id']).name != entry['paper_id'] or entry['paper_id'] in ('.', '..'):
            raise ValueError('Unsafe paper ID')
    cfg = Config.load(config_path)
    source_files = sources()
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    entries = [dict(entry, parsed_json_sha256=file_hash(entry.get('parsed_json')),
               parsed_assets_sha256=referenced_assets(entry.get('parsed_json')),
               ground_truth_sha256=file_hash(entry.get('ground_truth')),
               ocr_cache_path_sha256=file_hash(entry.get('ocr_cache_path')),
               qa_path_sha256=file_hash(entry.get('qa_path')))
               for entry in map(resolve_pdf_input, selected)]
    # Only this mkdir may create a run. Neither retries nor other commands overwrite it.
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen_manifest = run_dir / 'benchmark_manifest.json'
    if manifest_path:
        shutil.copy2(manifest_path, frozen_manifest)
    else:
        dump_json(frozen_manifest, manifest)
    shutil.copy2(config_path, run_dir / 'source_config.json')
    dump_json(run_dir / 'config.json', asdict(cfg))
    dump_json(run_dir / 'input_entries.json', entries)
    for path in source_files:
        target = run_dir / 'code_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    dump_json(run_dir / 'protocol.json', dict(
        version=VERSION, created_at=time.time(), manifest_sha256=sha256(frozen_manifest),
        input_entries_sha256=sha256(run_dir / 'input_entries.json'),
        source_config_sha256=sha256(config_path), code_hashes=hashes,
        selection=selection, entries=entries, full_benchmark=(len(entries) == len(manifest['papers'])
            and not papers and not limit and not ready_only),
        method='Source-verified metadata/pixel preflight -> bounded grounded understanding/draft/refinement -> two checked candidates, third only if both fail -> offline selection',
        input_policy=INPUT_POLICY, visual_policy='0/1/many; no minimum figure quota',
        completion_policy='verified_candidate_quality; only fully checked candidates eligible',
        candidate_limit=CANDIDATE_LIMIT, feedback_limit=2, layout_vlm_limit=2,
        metadata_policy='source_verified_authors', source_policy='whole_blocks',
        author_metadata_required=True,
        preflight_timeout_seconds=PREFLIGHT_TIMEOUT_SECONDS,
        preflight_attempt_limits_seconds=[PREFLIGHT_TIMEOUT_SECONDS*i for i in range(1,4)],
        policy='No retries of attempted papers; keep invalid inputs/failures in denominator; evaluation separate',
        attempt_definition='initial plus at most two feedback repairs; successful stages reused',
        third_layout_candidate='only_if_first_two_invalid',
        per_paper_max_calls=dict(understanding=3, draft=4, refinement=3, layout=3)))
    return selection


def frozen(run_dir):
    protocol = read(run_dir / 'protocol.json')
    if protocol['version'] != VERSION:
        raise ValueError('Unknown protocol version')
    if sha256(run_dir / 'benchmark_manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Benchmark manifest snapshot changed')
    if sha256(run_dir / 'source_config.json') != protocol['source_config_sha256']:
        raise ValueError('Config source snapshot changed')
    for name, digest in protocol['code_hashes'].items():
        if (sha256(ROOT / name) != digest or
                sha256(run_dir / 'code_snapshot' / name) != digest):
            raise ValueError(f'Frozen source differs: {name}; prepare a new run')
    if (sha256(run_dir / 'input_entries.json') != protocol['input_entries_sha256']
            or read(run_dir / 'input_entries.json') != protocol['entries']):
        raise ValueError('Frozen input resolution changed')
    cfg = Config.load(run_dir / 'config.json')
    if cfg.to_dict() != Config.load(run_dir / 'source_config.json').to_dict():
        raise ValueError('Effective configuration changed')
    return protocol, cfg


def stage_api(cfg, directory, label, calls, vlm=0):
    folder = directory / 'stages' / label
    ledger = BudgetLedger(replace(cfg.budget, max_calls=calls, max_vlm_calls=vlm), folder / 'cost.json')
    ledger.save()
    return APIClient(replace(cfg, budget=ledger.limits), ledger, folder / 'api_logs'), ledger


def verify_repair_candidate(story, decision, document, pool, cfg, ledger, directory, *, text_safety=None):
    dump_json(directory / 'decision.json', decision)
    dump_json(directory / 'checkpoint.json', dict(stage='evidence'))
    # Route from the same original order for both candidates, then apply only
    # the schema-checked order/preferences/text edits. No layout feedback can
    # add a figure, change a binding or replace the title/authors.
    routed, routing = reroute_visual_bindings(story, document, pool)
    ordered = apply_repair_decision(routed, decision)
    dump_json(directory / 'storyboard.json', ordered)
    errors, warnings = validate_provenance_storyboard(ordered, document)
    dump_json(directory / 'evidence_route.json', dict(status='failed' if errors else 'passed',
        routing=routing, warnings=warnings, errors=errors))
    if errors:
        raise ValueError(f'Grounded story invalid: {errors}')
    selected = {v['asset_id']: pool[v['asset_id']] for m in ordered['modules'] for v in m['visuals']}
    dump_json(directory / 'checkpoint.json', dict(stage='geometry'))
    layout, search = make_layout(ordered, document, selected, cfg.layout,
                                 preferences=decision['preferences'], text_safety=text_safety)
    dump_json(directory / 'layout.json', layout.to_dict())
    dump_json(directory / 'layout_search.json', search)
    geometry = audit_layout(ordered, layout, selected, cfg.layout)
    dump_json(directory / 'geometry_check.json', geometry)
    if geometry['hard_violations'] or geometry['missing_atoms']:
        raise LayoutGeometryError(geometry, search)
    dump_json(directory / 'checkpoint.json', dict(stage='render'))
    rendered = PosterRenderer(cfg.layout, ledger, text_safety=text_safety).render(layout, directory)
    dump_json(directory / 'checkpoint.json', dict(stage='glyph_inspection'))
    # PosterRenderer inspects glyphs in the converted PDF.
    dump_json(directory / 'render_checks.json', dict(defects=rendered['defects'],
                                                     valid=not rendered['defects']))
    if rendered['defects']:
        raise RenderedCandidateError(rendered['defects'], search)
    dump_json(directory / 'checkpoint.json', dict(stage='selection'))
    score = quality_score(layout, search, ordered, selected)
    dump_json(directory / 'quality_score.json', score)
    missing = [name for name in PRODUCTS if not (directory / name).is_file()]
    if missing:
        raise ValueError(f'Candidate products missing: {missing}')
    return score


def _feedback_targets(defects):
    return {d.get('target') for d in defects if isinstance(d, dict) and
            d.get('category') in ('rendered_textbox_overflow', 'rendered_missing_text')}


def _validate_feedback(proposal, story, defects, allow_edits):
    decision = validate_repair_decision(proposal, story, allow_content_edits=allow_edits)
    targets = _feedback_targets(defects)
    for edit in decision['content_edits']:
        section = edit['section_id'].removeprefix('module:')
        if f'{section}_b{edit["bullet_index"]}' not in targets:
            raise ValueError('Text compression must target an actual defective bullet')
    return decision


def _repair_decision(decision, story, iteration):
    """Safe local alternative after rejected/unchanged feedback; never drops evidence."""
    next_decision = deepcopy(decision)
    order = list(next_decision['module_order'])
    if len(order) > 1:
        order = order[1:] + order[:1]
    next_decision['module_order'] = order
    next_decision['preferences'] = {
        mid: dict(placement='top' if iteration == 0 else 'side', visual_priority=1)
        for mid in order}
    next_decision['reason'] = 'Bounded local repacking after failed or unchanged feedback'
    return next_decision


def _merge_feedback(proposal, story, decision, defects, allow_edits):
    # New edits refer to the *current* bullets. Keep previous accepted repairs.
    current = apply_repair_decision(story, decision)
    delta = _validate_feedback(proposal, current, defects, allow_edits)
    edits = {(e['section_id'], e['bullet_index']): deepcopy(e) for e in decision['content_edits']}
    edits.update({(e['section_id'], e['bullet_index']): e for e in delta['content_edits']})
    delta['content_edits'] = list(edits.values())
    return validate_repair_decision(delta, story, allow_content_edits=True)


def repair_poster(story, document, pool, cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dump_json(directory / 'baseline_storyboard.json', story)
    decision, history, text_safety = validate_repair_decision({}, story), [], {}
    records = [dict(id=f'candidate_{i}', status='pending', index=i,
                    directory=str(directory / f'candidate_{i}')) for i in range(CANDIDATE_LIMIT)]
    for row in records:
        folder = Path(row['directory'])
        folder.mkdir(exist_ok=False)
        for name in ('evidence_route.json', 'geometry_check.json', 'render_checks.json'):
            dump_json(folder / name, dict(status='not_run', valid=False))
        dump_json(folder / 'candidate_status.json', row)
    selection = dict(candidate_limit=CANDIDATE_LIMIT, feedback_limit=2, candidates=records,
                     selected=None, selection_policy=SELECTION_POLICY,
                     third_candidate_policy='only_if_first_two_invalid',
                     tie_break='earlier_candidate', ground_truth_used=False)
    dump_json(directory / 'candidate_selection.json', selection)
    api, ledger = stage_api(cfg, directory, 'layout', CANDIDATE_LIMIT, 2)
    api_available, last_rejection = True, None
    try:
        try:
            prompt = SCHEMA + '\nPAPER DATA:\n' + json.dumps(repair_context(story, pool), ensure_ascii=False)
            proposal = api.json('layout_plan', SYSTEM, prompt, max_tokens=1200)
            dump_json(directory / 'plan_response.json', proposal)
            decision = validate_repair_decision(proposal, story)
        except Exception as exc:
            last_rejection = failure_feedback(exc, api)
            api_available = last_rejection['retryable']
            history.append(dict(stage='plan', status='rejected', feedback=last_rejection, fallback='baseline'))
        for iteration, row in enumerate(records):
            folder = Path(row['directory'])
            if iteration == 2 and any(r['status'] == 'valid' for r in records[:2]):
                row.update(status='skipped', reason='valid_candidate_already_available')
                dump_json(folder / 'candidate_status.json', row)
                break
            failure = None
            row.update(status='running', decision=deepcopy(decision), text_safety=dict(text_safety))
            dump_json(folder / 'candidate_status.json', row)
            try:
                score = verify_repair_candidate(story, decision, document, pool, cfg, ledger, folder,
                                         text_safety=text_safety)
                row.update(status='valid', quality_score=score['score'])
            except Exception as exc:
                failure = exc
                checkpoint = folder / 'checkpoint.json'
                failed_stage = read(checkpoint)['stage'] if checkpoint.is_file() else 'candidate'
                row.update(status='failed', failed_stage=failed_stage,
                           **failure_feedback(exc), defects=getattr(exc, 'defects', []))
                dump_json(folder / 'failure.json', dict(row,
                    artifacts=[str(p.relative_to(folder)) for p in sorted(folder.iterdir()) if p.is_file()]))
                if read(folder / 'render_checks.json').get('status') == 'not_run':
                    dump_json(folder / 'render_checks.json', dict(status='not_run', valid=False,
                        blocked_by=failed_stage, error_type=type(exc).__name__))
            row['artifacts'] = [str(p.relative_to(folder)) for p in sorted(folder.iterdir()) if p.is_file()]
            dump_json(folder / 'candidate_status.json', row)
            history.append(dict(stage=row['id'], **{k: v for k, v in row.items() if k != 'index'}))
            dump_json(directory / 'candidate_selection.json', selection)
            if iteration == CANDIDATE_LIMIT - 1 or (iteration == 1 and any(r['status'] == 'valid' for r in records)):
                continue
            trigger = ('post_render_inspection' if isinstance(failure, RenderedCandidateError) else
                       'layout_geometry' if isinstance(failure, LayoutGeometryError) else
                       'candidate_error' if failure else 'visual_review')
            defects = getattr(failure, 'defects', [])
            for target in _feedback_targets(defects):
                if target:
                    text_safety[target] = min(2, text_safety.get(target, 0) + 1)
            feedback_data = dict(attempt=iteration+2, attempt_limit=3, trigger=trigger, defects=defects,
                failed_stage=row.get('failed_stage'), error_type=row.get('error_type'),
                validation_error=row.get('validation_error'), diagnostic=row.get('diagnostic'),
                geometry=getattr(failure, 'geometry', None), current_decision=decision,
                local_text_safety=text_safety, prior_feedback_rejection=last_rejection,
                previous_attempts=[{k:r.get(k) for k in ('id','status','failed_stage','defects')} for r in records[:iteration+1]],
                instructions='Repair the specific failed checks, not an identical rerun. '
                'Preserve scientific scope, negation, numbers, source_ids and figure bindings. '
                'Only an actual defective bullet may be shortened; do not alter title/authors/captions. '
                'A layout-only or runtime failure permits only order/placement/priority changes. '
                'Return NEW edits only; previously accepted edits are already applied in DATA. '
                'Never claim source raster label legibility was measured.')
            fb_folder = directory / 'feedback_attempts' / f'feedback_{iteration+1}'
            dump_json(fb_folder / 'request.json', feedback_data)
            try:
                if not api_available:
                    raise ValueError('Feedback API unavailable after permanent error; local recovery only')
                search_path = folder / 'layout_search.json'
                search = read(search_path) if search_path.is_file() else None
                feedback = api.json(f'visual_feedback_{iteration+1}', SYSTEM,
                    json.dumps(feedback_data, ensure_ascii=False) + '\n' + SCHEMA + '\nDATA:\n' +
                    json.dumps(repair_context(apply_repair_decision(story, decision), pool, search), ensure_ascii=False),
                    images=[folder / 'poster.png'] if (folder / 'poster.png').is_file() else (),
                    max_tokens=1800)
                dump_json(fb_folder / 'response.json', feedback)
                revised = _merge_feedback(feedback, story, decision, defects,
                    allow_edits=isinstance(failure, RenderedCandidateError))
                same = all(revised[k] == decision[k] for k in ('module_order','preferences','content_edits'))
                decision = _repair_decision(revised, story, iteration) if failure and same else revised
                last_rejection = None
                event = dict(stage=f'visual_feedback_{iteration+1}', status='accepted', trigger=trigger,
                             unchanged_response=same, local_repacking=bool(failure and same))
            except Exception as exc:
                last_rejection = failure_feedback(exc, api)
                api_available = api_available and last_rejection['retryable']
                decision = _repair_decision(decision, story, iteration) if failure else decision
                event = dict(stage=f'visual_feedback_{iteration+1}', status='rejected', trigger=trigger,
                             feedback=last_rejection, fallback='local_repacking' if failure else 'incumbent')
            dump_json(fb_folder / 'status.json', event)
            history.append(event)
            dump_json(directory / 'history.json', history)
        valid = [row for row in records if row['status'] == 'valid']
        if not valid:
            raise ValueError('No valid rendered candidate after three attempts; all diagnostics retained')
        chosen = max(valid, key=lambda row: (row['quality_score'], -row['index']))
        for name in PRODUCTS:
            shutil.copy2(Path(chosen['directory']) / name, directory / name)
        selection['selected'] = chosen['id']
        return dict(valid_candidates=len(valid), chosen=chosen['id'],
                    selection_policy=SELECTION_POLICY, candidate_limit=CANDIDATE_LIMIT,
                    candidates_attempted=sum(r['status'] in ('valid','failed') for r in records))
    finally:
        dump_json(directory / 'candidate_selection.json', selection)
        dump_json(directory / 'history.json', history)
        ledger.save()


def select_evidence_with_audit(raw, api, cfg, directory):
    """Keep verified metadata immutable; repair only failed content understanding."""
    from dagent.core.content.source_metadata import required_author_audit
    directory = Path(directory)
    locked = required_author_audit(raw)
    if not locked['valid']:
        raise ValueError('Preflight must supply source-verified authors: ' + ', '.join(locked['errors']))
    history = []
    for attempt in range(ATTEMPT_LIMIT):
        doc = deepcopy(raw)  # never leak a partially mutated failed attempt
        folder = directory / 'understanding_attempts' / f'attempt_{attempt + 1}'
        try:
            doc = select_evidence_pointers(doc, FeedbackAPI(api, attempt, history[-1] if history else {}), cfg)
            dump_json(folder / 'model_author_proposal_audit.json', doc.parsing.get('author_metadata_audit', {}))
            doc.parsing['author_metadata_audit'] = deepcopy(raw.parsing['author_metadata_audit'])
            doc.authors = raw.authors
            if not any(not c.asset_id and c.source_ids for c in doc.candidates):
                raise ValueError('No usable source-grounded text nodes; rejected nodes: ' +
                                 json.dumps(doc.parsing.get('rejected_text_nodes', []))[:1000])
            audit = required_author_audit(doc)
            if not audit['valid']:
                raise ValueError('Locked author metadata invalid: ' + ', '.join(audit['errors']))
            history.append(dict(attempt=attempt+1, status='valid'))
            dump_json(directory / 'required_author_metadata.json', audit)
            return doc
        except Exception as exc:
            failure = failure_feedback(exc, api)
            history.append(dict(attempt=attempt+1, status='failed', **failure))
            if not failure['retryable'] or attempt == ATTEMPT_LIMIT - 1:
                raise
        finally:
            proposed_title = doc.title
            doc.title = raw.title
            title_audit = dict(raw.parsing.get('title_selection_audit', {}),
                              understanding_proposed_title=proposed_title,
                              selected_title=raw.title, title_locked=True, authors_locked=True)
            dump_json(directory / 'title_selection_audit.json', title_audit)
            for key in ('understanding_source_audit', 'author_metadata_audit'):
                value = doc.parsing.get(key, dict(status='not_available'))
                dump_json(folder / (key + '.json'), value)
                dump_json(directory / (key + '.json'), value)
            dump_json(folder / 'status.json', history[-1])
            dump_json(directory / 'understanding_history.json', history)


def generate_one(entry, cfg, directory, prepared_input=None):
    paper_id = entry['paper_id']
    directory.mkdir(parents=True, exist_ok=False)
    dump_json(directory / 'status.json', dict(status='running', paper_id=paper_id,
        pdf_sha256=entry['pdf_sha256'], simulated_api=False, started_at=time.time()))
    stage = 'input'
    try:
        check_input(entry)
        stage = 'preflight'
        if prepared_input is None:
            prepared_input = directory / 'input_preflight'
            prepare_evidence_input(entry, cfg, prepared_input)
        raw, pool = load_preflight(prepared_input)
        if raw.pdf_sha256 != entry['pdf_sha256'] or raw.paper_id != paper_id:
            raise InputFailure('preflight_identity_mismatch',
                               'Offline preflight belongs to a different paper')
        dump_json(directory / 'parsed_document.json', raw.to_dict())
        dump_json(directory / 'prepared_assets.json', pool)
        api, ledger = stage_api(cfg, directory, 'understanding', ATTEMPT_LIMIT)
        try:
            stage = 'understanding'
            understood = select_evidence_with_audit(raw, api, cfg, directory)
            doc = understood.to_dict()
            dump_json(directory / 'understood_document.json', doc)
        finally:
            ledger.save()
        api, ledger = stage_api(cfg, directory, 'draft', 4)
        try:
            stage = 'draft'
            try:
                notes = construct_provenance_storyboard(doc, api, directory / 'draft')
            except ValueError:
                # The revision agent may still use the last complete grounded attempt.
                candidates = sorted((directory / 'draft').glob('candidate_*.json'))
                if not candidates:
                    raise
                usable = [(path, read(path)) for path in reversed(candidates)]
                usable = [(path, value) for path, value in usable
                          if isinstance(value, dict) and isinstance(value.get('modules'), list)]
                if not usable:
                    raise
                chosen, notes = usable[0]
                dump_json(directory / 'draft_fallback.json', dict(candidate=chosen.name,
                    reason='Initial draft had no valid complete candidate; attempted grounded refinement'))
        finally:
            ledger.save()
        api, ledger = stage_api(cfg, directory, 'refinement', ATTEMPT_LIMIT)
        try:
            stage = 'refinement'
            story, checked = revise_grounded_storyboard(doc, notes, pool, api, directory / 'refinement')
        finally:
            ledger.save()
        stage = 'layout'
        result = repair_poster(story, checked, pool, cfg, directory)
        if sha256(Path(entry['pdf'])) != entry['pdf_sha256']:
            raise ValueError('PDF changed during generation')
        dump_json(directory / 'status.json', dict(status='complete', paper_id=paper_id,
            pdf_sha256=entry['pdf_sha256'], simulated_api=False, completed_at=time.time(),
            visual_mode=visual_mode(len(pool)), input_recovery=entry.get('input_recovery', {}).get('status'), **result))
    except Exception as exc:
        dump_json(directory / 'status.json', dict(status='failed', paper_id=paper_id,
            pdf_sha256=entry['pdf_sha256'],
            failed_stage=exc.stage if isinstance(exc, InputFailure) else stage,
            error_type=type(exc).__name__, diagnostic=safe_error_metadata(exc),
            failure_code=exc.code if isinstance(exc, InputFailure) else None,
            reason=str(exc)[:1600] if isinstance(exc, ValueError) else None,
            failure_feedback=failure_feedback(exc)))


def report(run_dir):
    from evaluation.run import summarize
    from dagent.core.evaluation.records import evaluation_failed
    protocol = read(run_dir / 'protocol.json')
    rows = []
    evaluation_rows = []
    for entry in protocol['entries']:
        directory = run_dir / 'generation' / entry['paper_id']
        status = read(directory / 'status.json') if (directory / 'status.json').exists() else {'status': 'pending'}
        costs = {}
        for stage in ('understanding', 'draft', 'refinement', 'layout'):
            path = directory / 'stages' / stage / 'cost.json'
            ledger = read(path) if path.is_file() else {}
            costs[stage] = {key: ledger.get(key, 0 if key != 'usd' else None)
                            for key in ('calls', 'total_tokens', 'usd')}
        evaluation = directory / 'evaluation' / 'cost.json'
        ledger = read(evaluation) if evaluation.is_file() else {}
        costs['evaluation'] = {key: ledger.get(key, 0 if key != 'usd' else None)
                               for key in ('calls', 'total_tokens', 'usd')}
        result_path = directory / 'evaluation' / 'metrics.json'
        if not result_path.is_file():
            result_path = directory / 'evaluation_result.json'
        evaluation_result = read(result_path) if result_path.is_file() else None
        if evaluation_result is not None and evaluation_result.get('status') == 'complete':
            evaluation_state = 'incomplete' if evaluation_failed(evaluation_result) else 'complete'
        elif (directory / 'evaluation_error.json').is_file() or evaluation_result is not None:
            evaluation_state = 'failed'
        elif (directory / 'evaluation_started.json').is_file():
            evaluation_state = 'interrupted'
        else:
            evaluation_state = 'pending'
        evaluation_rows.append(dict(paper_id=entry['paper_id'], result=evaluation_result,
            status='skipped' if evaluation_result is None else
                   'fail' if evaluation_failed(evaluation_result) else 'ok'))
        preflight_path = run_dir / 'input_preflight' / entry['paper_id'] / 'status.json'
        preflight = read(preflight_path) if preflight_path.is_file() else {'status': 'pending'}
        rows.append(dict(paper_id=entry['paper_id'], input_status=entry.get('input_status'),
                         preflight=preflight,
                         status=status, evaluation_status=evaluation_state, costs=costs))
    totals = {}
    for stage in ('understanding', 'draft', 'refinement', 'layout', 'evaluation'):
        values = [row['costs'][stage] for row in rows]
        totals[stage] = dict(calls=sum(v['calls'] for v in values),
                             total_tokens=sum(v['total_tokens'] for v in values),
                             usd=sum(v['usd'] for v in values) if all(v['usd'] is not None for v in values) else None)
    states = {state: sum(r['status']['status'] == state for r in rows)
              for state in ('complete', 'failed', 'running', 'pending')}
    evaluation_states = {state: sum(r['evaluation_status'] == state for r in rows)
                         for state in ('complete', 'incomplete', 'failed', 'interrupted', 'pending')}
    evaluation_summary = summarize(evaluation_rows)
    dump_json(run_dir / 'evaluation_summary.json', evaluation_summary)
    summary = dict(version=VERSION, full_benchmark=protocol['full_benchmark'],
                   denominator=len(rows), states=states, evaluation_states=evaluation_states,
                   preflight_states={s: sum(r['preflight']['status'] == s for r in rows)
                                     for s in ('ready', 'failed', 'running', 'pending')},
                   success_rate=states['complete'] / len(rows),
                   visual_modes={mode: sum(r['preflight'].get('visual_mode') == mode for r in rows)
                                 for mode in ('rich_visual', 'single_visual', 'text_only')},
                   recovered_inputs=sum(e.get('input_recovery', {}).get('status') == 'recovered'
                                        for e in protocol['entries']),
                   generation_calls=sum(totals[k]['calls'] for k in ('understanding', 'draft', 'refinement', 'layout')),
                   costs=totals, evaluation=evaluation_summary, rows=rows)
    dump_json(run_dir / 'summary.json', summary)
    return summary



def ensure_preflight(run_dir, entry):
    """Three isolated local attempts; timeout recovery changes the time budget.

    Permanent invalid input is not repaired by rerunning it. Metadata already has
    its own three source strategies. No API calls occur in any preflight worker.
    """
    directory = run_dir / 'input_preflight' / entry['paper_id']
    if directory.exists():
        return directory  # attempted runs are immutable; use a new run to retry
    directory.mkdir(parents=True, exist_ok=False)
    try:
        check_input(entry)
    except Exception as exc:
        write_input_failure(directory, exc, 'input')
        return directory
    history = []
    for attempt in range(1, ATTEMPT_LIMIT + 1):
        folder = directory / 'attempts' / f'attempt_{attempt}'
        folder.mkdir(parents=True, exist_ok=False)
        timeout = PREFLIGHT_TIMEOUT_SECONDS * attempt
        request = dict(attempt=attempt, attempt_limit=ATTEMPT_LIMIT, timeout_seconds=timeout,
                       feedback=history[-1] if history else None,
                       strategy='isolated_worker' if attempt==1 else 'fresh_worker_with_larger_time_budget',
                       checks_unchanged=True, api_calls=0)
        dump_json(folder / 'attempt_request.json', request)
        dump_json(folder / 'status.json', dict(status='running', stage='document', paper_id=entry['paper_id']))
        dump_json(directory / 'status.json', dict(status='running', stage='preflight', attempt=attempt, paper_id=entry['paper_id']))
        start = time.time()
        with (folder / 'extraction.log').open('w') as log:
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '_preflight-worker', '--run-dir', str(run_dir), '--paper', entry['paper_id'],
                    '--attempt', str(attempt)], stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False)
                status = read(folder / 'status.json')
                if result.returncode != 0 or status.get('status') == 'running':
                    write_input_failure(folder, InputFailure('input_worker_failed',
                        f'Offline PDF worker did not complete (exit {result.returncode})'), 'document')
            except subprocess.TimeoutExpired:
                write_input_failure(folder, InputFailure('input_preflight_timeout',
                    f'PDF preflight exceeded attempt {attempt} budget of {timeout} seconds'), 'document')
        status = read(folder / 'status.json')
        history.append(dict(attempt=attempt, timeout_seconds=timeout, wall_seconds=time.time()-start,
                            **status))
        dump_json(directory / 'attempt_history.json', history)
        if status['status'] == 'ready':
            # No copies of potentially huge assets; the chosen immutable attempt
            # and its status/artifact hashes form the prepared-input envelope.
            dump_json(directory / 'status.json', dict(status,
                prepared_directory=str(folder.relative_to(directory)),
                prepared_status_sha256=sha256(folder / 'status.json'),
                attempts_used=attempt, attempt_limit=ATTEMPT_LIMIT))
            return directory
        retryable = status.get('failure_code') in ('input_preflight_timeout', 'input_worker_failed')
        if not retryable or attempt == ATTEMPT_LIMIT:
            dump_json(directory / 'status.json', dict(status, attempts_used=attempt, attempt_limit=ATTEMPT_LIMIT))
            return directory
    return directory


def preflight_run(run_dir, workers=2):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    run_dir = writable_path(run_dir)
    with run_lock(run_dir):
        protocol, _ = frozen(run_dir)
        with ThreadPoolExecutor(max_workers=max(1,min(4,workers))) as executor:
            pending={executor.submit(ensure_preflight,run_dir,e):e for e in protocol['entries']}
            for future in as_completed(pending):
                entry=pending[future]; directory=future.result()
                status=read(directory/'status.json')
                print(entry['paper_id'],status['status'],status.get('failure_code',''),flush=True)
                report(run_dir)
        return report(run_dir)


def preflight_worker(run_dir, paper, attempt=1):
    # Parent holds the run lock and has verified the frozen protocol.
    run_dir = writable_path(run_dir)
    protocol = read(run_dir / 'protocol.json')
    entry = next(e for e in protocol['entries'] if e['paper_id'] == paper)
    directory = run_dir / 'input_preflight' / paper / 'attempts' / f'attempt_{attempt}'
    stage = 'input'
    try:
        check_input(entry)
        stage = 'document'
        prepare_evidence_input(entry, Config.load(run_dir / 'config.json'), directory)
        check_input(entry)
    except Exception as exc:
        write_input_failure(directory, exc, stage)


def run(run_dir):
    run_dir = writable_path(run_dir)
    with run_lock(run_dir):
        protocol, cfg = frozen(run_dir)
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env', override=False)
        for entry in protocol['entries']:
            folder = run_dir / 'generation' / entry['paper_id']
            if folder.exists():
                continue  # Includes interrupted/failed attempts: never silently rebill.
            prepared = ensure_preflight(run_dir, entry)
            generate_one(entry, cfg, folder, prepared_input=prepared)
            summary = report(run_dir)
            status = read(folder / 'status.json')
            print(entry['paper_id'], status['status'],
                  status.get('failed_stage', ''), status.get('failure_code') or status.get('error_type', ''),
                  f"{summary['states']['complete']}/{summary['denominator']}", flush=True)
        return report(run_dir)


def evaluate_run(run_dir, metrics):
    run_dir = writable_path(run_dir)
    with run_lock(run_dir):
        protocol, cfg = frozen(run_dir)
        from dotenv import load_dotenv
        from dagent.core.evaluation.benchmark_eval import evaluate
        from dagent.core.evaluation.records import normalize_metrics
        load_dotenv(ROOT / '.env', override=False)
        metrics = normalize_metrics(metrics)
        for entry in protocol['entries']:
            folder = run_dir / 'generation' / entry['paper_id']
            if not (folder / 'status.json').is_file() or read(folder / 'status.json')['status'] != 'complete':
                continue
            marker = folder / 'evaluation_started.json'
            if marker.exists():
                continue  # Prevent silent retries even after interrupted paid evaluation.
            dump_json(marker, dict(metrics=metrics, started_at=time.time()))
            try:
                check_input(entry)
                check_evaluation_input(entry)
                result = evaluate(folder, entry, cfg, metrics=metrics)
                dump_json(folder / 'evaluation_result.json', result)
            except Exception as exc:
                dump_json(folder / 'evaluation_error.json', dict(error_type=type(exc).__name__,
                    diagnostic=safe_error_metadata(exc)))
            report(run_dir)
            print(entry['paper_id'], 'evaluated' if (folder / 'evaluation_result.json').exists()
                  else 'evaluation_failed', flush=True)
        return report(run_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Offline: freeze inputs, config, and code')
    prep.add_argument('--run-dir', required=True)
    prep.add_argument('--manifest', default=DEFAULT_MANIFEST,
                      help='Explicit manifest override; otherwise inventory --benchmark-root/papers')
    prep.add_argument('--benchmark-root', default=str(DEFAULT_BENCHMARK_ROOT))
    prep.add_argument('--config', default=str(DEFAULT_CONFIG))
    prep.add_argument('--papers', nargs='+')
    prep.add_argument('--limit', type=int, default=0)
    prep.add_argument('--ready-only', action='store_true')
    for name, flag in [('run', '--allow-paid-generation'), ('evaluate', '--allow-paid-evaluation')]:
        command = commands.add_parser(name)
        command.add_argument('--run-dir', required=True)
        command.add_argument(flag, action='store_true', required=True)
        if name == 'evaluate':
            command.add_argument('--metrics', nargs='+', choices=['judge', 'qa'], default=['judge', 'qa'])
    check = commands.add_parser('preflight', help='Offline: extract and validate all inputs, no API calls')
    check.add_argument('--run-dir', required=True)
    check.add_argument('--workers', type=int, default=2)
    worker = commands.add_parser('_preflight-worker', help=argparse.SUPPRESS)
    worker.add_argument('--run-dir', required=True)
    worker.add_argument('--paper', required=True)
    worker.add_argument('--attempt', type=int, choices=(1,2,3), default=1)
    rep = commands.add_parser('report', help='Offline: summarize all selected papers')
    rep.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    if args.command == '_preflight-worker':
        preflight_worker(args.run_dir, args.paper, args.attempt)
        return
    if args.command == 'prepare':
        result = prepare(args.run_dir, args.manifest, args.config,
                         papers=args.papers, limit=args.limit, ready_only=args.ready_only,
                         benchmark_root=args.benchmark_root)
    elif args.command == 'preflight':
        result = preflight_run(args.run_dir, args.workers)
    elif args.command == 'run':
        result = run(args.run_dir)
    elif args.command == 'evaluate':
        result = evaluate_run(args.run_dir, args.metrics)
    else:
        result = report(writable_path(args.run_dir))
    print(json.dumps(result if args.command == 'prepare' else
        {k: result[k] for k in ('denominator', 'states', 'preflight_states', 'evaluation_states',
                               'visual_modes', 'recovered_inputs', 'generation_calls', 'success_rate', 'costs', 'full_benchmark')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
