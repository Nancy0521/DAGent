"""Offline-only input compatibility and crop diagnostics for DAGent.

Keep cached parser text, but never relabel cached pixels as verified PDF crops.
All poster images are freshly sampled from the hash-matched original PDF.
"""
import json
from pathlib import Path
import re

from dagent.core.config import dump_json
from dagent.core.document import EvidenceCandidate, SourceDocument, read_mineru, sha256
from dagent.core.layout.asset_preparation import prepare_assets
from dagent.agents.native import read_pdf, pixel_audit
from dagent.agents.titles import select_title
from dagent.agents.metadata import recover_authors

INPUT_POLICY = 'dagent_native_input_validation'


def visual_mode(count):
    return 'rich_visual' if count >= 2 else 'single_visual' if count == 1 else 'text_only'


class InputFailure(ValueError):
    def __init__(self, code, reason, stage='preflight'):
        super().__init__(reason)
        self.code = code
        self.stage = stage


def write_input_failure(directory, exc, stage):
    """Only called on local input operations, never provider exceptions."""
    dump_json(Path(directory) / 'status.json', dict(
        status='failed', stage=exc.stage if isinstance(exc, InputFailure) else stage,
        failure_code=exc.code if isinstance(exc, InputFailure) else 'local_input_error',
        error_type=type(exc).__name__, reason=str(exc)[:500], api_calls=0))


def load_source_document(entry, output):
    import fitz
    output = Path(output)
    pdf = Path(entry['pdf'])
    with fitz.open(pdf) as document:
        if document.needs_pass:
            raise InputFailure('password_required',
                               'Source PDF requires a password', 'document')
        if not len(document):
            raise InputFailure('invalid_pdf_no_pages',
                               'Source PDF contains no pages', 'document')
    parsed = entry.get('parsed_json')
    diagnostics, table_audit, previous = [], [], []
    blocks, native_assets, metadata_title = read_pdf(pdf, output / 'native_assets', diagnostics)
    native_blocks = blocks
    cached_text_preserved = False
    if parsed:
        try:
            # Cached pixels are never decoded: even a corrupt unused cache image
            # must not veto otherwise valid parser text and native source crops.
            cached, previous = read_mineru(Path(parsed), table_audit=table_audit)
            if sum(len(b['text']) for b in cached) >= 100:
                blocks = cached
                cached_text_preserved = True
            else:
                diagnostics.append(dict(stage='cached_text', reason='insufficient_text_use_native'))
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            diagnostics.append(dict(stage='cached_text', error_type=type(exc).__name__,
                                    reason='unreadable_cache_use_native'))
    dump_json(output / 'extraction_diagnostics.json', diagnostics)
    if not blocks or sum(len(b['text']) for b in blocks) < 100:
        raise InputFailure('no_usable_source_text',
                           'Neither cached nor native source text is usable', 'document')
    end = next((i for i, b in enumerate(blocks) if i > len(blocks)//3 and
                re.fullmatch(r'(?:\d+[. ]*)?(References|Bibliography)', b['text'].strip(), re.I)), len(blocks))
    blocks = blocks[:end]
    title, title_audit = select_title(entry, metadata_title, blocks, native_blocks)
    dump_json(output / 'title_selection_audit.json', title_audit)
    raw = SourceDocument(entry['paper_id'], str(pdf.resolve()), sha256(pdf),
        title, '', blocks, native_assets,
        parsing=dict(backend='existing_mineru_text_native_pdf_visuals' if cached_text_preserved else 'pymupdf_native',
                     parsed_json=parsed, table_audit=table_audit, local_inference=False,
                     asset_crops_require_visual_review=True, benchmark_input_policy=INPUT_POLICY,
                     input_recovery=entry.get('input_recovery'), extraction_warnings=diagnostics,
                     title_selection_audit=title_audit))
    dump_json(output / 'input_adapter.json', dict(
        policy=INPUT_POLICY, pdf_sha256=raw.pdf_sha256, parsed_json=parsed,
        cached_text_preserved=cached_text_preserved, cached_images_used=False,
        input_recovery=entry.get('input_recovery'), extraction_warnings=diagnostics,
        previous_assets=[dict(id=a['id'], caption=a.get('caption'),
                              path=a['path']) for a in previous],
        native_asset_ids=[a['id'] for a in raw.assets],
        asset_ids_resolved_before_any_model_call=True))
    return raw


def verify_visual_assets(document, output, cfg):
    """Exclude unsafe crops, not whole papers. Zero verified images is supported."""
    import fitz
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference_pages = [b['page'] for b in document['blocks'] if re.fullmatch(
        r'(?:\d+[.]?\s+)?references', b.get('text', '').strip(), re.I)]
    cutoff = min(reference_pages) if reference_pages else float('inf')
    candidates = [a for a in document['assets'] if a['page'] <= cutoff][:12]
    prepared, records, boxes = {}, [], []
    for a in candidates:
        folder = output / a['id']
        story = dict(modules=[dict(id='preflight', visuals=[
            dict(asset_id=a['id'], source_ids=['asset:' + a['id']])])])
        try:
            value = prepare_assets(story, document, folder, cfg)[a['id']]
            pixels = pixel_audit(value['path'])
            dump_json(folder / 'pixel_audit.json', pixels)
            if not pixels['valid']:
                raise ValueError('Prepared crop failed pixel gate: ' + ', '.join(pixels['failures']))
            value['pixel_audit'] = pixels
            audit = json.loads((folder / 'audit.json').read_text())[0]
            crop = audit['source_crop']
            box = fitz.Rect(crop['bbox'])
            duplicate = next((aid for aid, page, prior in boxes
                if page == crop['page'] and (box & prior).get_area() /
                max(1, min(box.get_area(), prior.get_area())) > .08), None)
            if duplicate:
                records.append(dict(asset_id=a['id'], status='excluded_overlap',
                                    overlaps=duplicate))
                continue
            boxes.append((a['id'], crop['page'], box))
            prepared[a['id']] = value
            records.append(dict(asset_id=a['id'], status='available',
                                audit_path=str(folder / 'audit.json'),
                                pixel_audit=pixels, source_crop=a.get('source_crop'),
                                figure_aspect=value['aspect']))
        except Exception as exc:
            records.append(dict(asset_id=a['id'], status='excluded_crop_error',
                                error_type=type(exc).__name__, reason=str(exc)[:250],
                                source_crop=a.get('source_crop')))
    dump_json(output / 'inventory.json', dict(
        main_text_cutoff_page=cutoff if cutoff != float('inf') else None,
        candidate_limit=12, total_source_assets=len(document['assets']),
        assets=records, available_ids=list(prepared),
        minimum_required=0, visual_quota_unchanged=False,
        visual_mode=visual_mode(len(prepared)),
        warnings=['source_visuals_unavailable_use_grounded_text'] if not prepared else
                 ['single_source_visual_no_repetition'] if len(prepared) == 1 else []))
    dump_json(output / 'prepared_assets.json', prepared)
    return prepared


def prepare_evidence_input(entry, cfg, directory):
    directory = Path(directory)
    try:
        raw = load_source_document(entry, directory / 'source')
    except Exception as exc:
        write_input_failure(directory, exc, 'document')
        raise
    try:
        author_audit = recover_authors(raw, directory)
    except Exception as exc:
        failure = InputFailure('author_metadata_unverified', str(exc), 'metadata')
        write_input_failure(directory, failure, 'metadata')
        raise failure from exc
    dump_json(directory / 'parsed_document.json', raw.to_dict())
    try:
        pool = verify_visual_assets(raw.to_dict(), directory / 'preflight', cfg.layout)
    except Exception as exc:
        write_input_failure(directory, exc, 'preflight')
        raise
    dump_json(directory / 'prepared_assets.json', pool)
    # Hash every local preflight artifact before any paid stage consumes it.
    hashes = {str(p.relative_to(directory)): sha256(p)
              for p in sorted(directory.rglob('*')) if p.is_file()
              and p.name not in ('status.json', 'extraction.log', 'artifacts.json')}
    dump_json(directory / 'artifacts.json', dict(input_policy=INPUT_POLICY, files=hashes))
    dump_json(directory / 'status.json', dict(status='ready', stage='preflight',
        paper_id=entry['paper_id'], available_visuals=len(pool),
        author_metadata_valid=True, metadata_attempts=author_audit['attempts_used'],
        visual_mode=visual_mode(len(pool)), input_recovery=entry.get('input_recovery', {}).get('status'),
        artifacts_sha256=sha256(directory / 'artifacts.json'), api_calls=0))
    return raw, pool


def active_preflight_directory(directory):
    directory = Path(directory)
    status_path = directory / 'status.json'
    if not status_path.is_file():
        return directory
    status = json.loads(status_path.read_text())
    if status.get('status') != 'ready' or 'prepared_directory' not in status:
        return directory
    chosen = (directory / status['prepared_directory']).resolve()
    if not chosen.is_relative_to(directory.resolve()) or chosen == directory.resolve():
        raise InputFailure('preflight_changed', 'Unsafe prepared attempt path')
    child = chosen / 'status.json'
    if not child.is_file() or sha256(child) != status.get('prepared_status_sha256'):
        raise InputFailure('preflight_changed', 'Chosen preflight attempt changed')
    return chosen


def check_preflight(directory):
    directory = active_preflight_directory(directory)
    status_path = directory / 'status.json'
    if not status_path.is_file():
        raise InputFailure('incomplete_preflight', 'Offline preflight has no completion record')
    status = json.loads(status_path.read_text())
    if status.get('status') != 'ready':
        raise InputFailure(status.get('failure_code', 'incomplete_preflight'),
                           status.get('reason', 'Offline preflight did not finish'),
                           status.get('stage', 'preflight'))
    manifest = directory / 'artifacts.json'
    if not manifest.is_file() or sha256(manifest) != status.get('artifacts_sha256'):
        raise InputFailure('preflight_changed', 'Offline artifact manifest changed or is missing')
    hashes = json.loads(manifest.read_text())
    if hashes.get('input_policy') != INPUT_POLICY:
        raise InputFailure('preflight_changed', 'Offline input policy changed')
    for name, digest in hashes['files'].items():
        path = (directory / name).resolve()
        if (not path.is_relative_to(directory.resolve()) or not path.is_file()
                or sha256(path) != digest):
            raise InputFailure('preflight_changed', f'Offline input artifact changed: {name}')


def load_preflight(directory):
    directory = active_preflight_directory(directory)
    check_preflight(directory)
    data = json.loads((directory / 'parsed_document.json').read_text())
    data['candidates'] = [EvidenceCandidate(**c) for c in data.get('candidates', [])]
    raw = SourceDocument(**data)
    pool = json.loads((directory / 'prepared_assets.json').read_text())
    # Do not ask paid planners to choose crops already excluded locally.
    raw.assets = [a for a in raw.assets if a['id'] in pool]
    raw.parsing['visual_mode'] = visual_mode(len(pool))
    return raw, pool
