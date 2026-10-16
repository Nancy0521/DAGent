"""Read-only, local PDF recovery performed ONLY before freezing a new run.

A matching directory alone is insufficient: a replacement must contain the
normalized paper title on page one. No benchmark poster/QA is read for recovery.
"""
from copy import deepcopy
from pathlib import Path
import unicodedata

from dagent.core.document import sha256

POLICY = 'source_pdf_recovery'


def normalized_title(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    return ''.join(c for c in value if c.isalnum())


def probe_pdf(path):
    import fitz
    path = Path(path)
    result = dict(path=str(path.resolve()), sha256=sha256(path) if path.is_file() else None)
    try:
        with fitz.open(path) as doc:
            if doc.needs_pass:
                return dict(result, readable=False, reason='password_required', pages=None)
            if not len(doc):
                return dict(result, readable=False, reason='invalid_pdf_no_pages', pages=0)
            # Force page-tree access; a nonzero page_count alone is insufficient.
            first = doc[0].get_text()
            for page in doc:
                _ = page.rect
            return dict(result, readable=True, reason=None, pages=len(doc), first_page_text=first)
    except (ValueError, RuntimeError, OSError) as exc:
        return dict(result, readable=False, reason='invalid_pdf_unreadable',
                    pages=None, error_type=type(exc).__name__)


def candidate_paths(entry):
    """Only known sibling parser roots; never glob arbitrary PDFs by similarity."""
    original = Path(entry['pdf']).resolve()
    if original.parent.name != entry['paper_id']:
        return []
    root = original.parents[2]
    candidates = []
    for parser_root in sorted((root / 'paper2poster').glob('parsed_papers*')):
        folder = parser_root / entry['paper_id']
        candidates.extend([folder / (entry['paper_id'] + '.pdf'),
                           *sorted(folder.glob('*_origin.pdf'))])
    return list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()
                             and p.resolve() != original))


def resolve_pdf_input(entry):
    result = deepcopy(entry)
    original = probe_pdf(entry['pdf'])
    expected = entry.get('pdf_sha256')
    title = next((line.removeprefix('title=') for line in
                  (entry.get('source_metadata') or '').splitlines() if line.startswith('title=')),
                 entry['paper_id'].replace('_', ' '))
    original_record = {k: v for k, v in original.items() if k != 'first_page_text'}
    audit = dict(policy=POLICY, status='original', original=original_record,
                 manifest_pdf_sha256=expected, expected_title=title, attempts=[],
                 resolved=dict(original_record), reason=original['reason'])
    result['input_recovery'] = audit
    # Changed input is an integrity error, NOT permission to silently accept new bytes.
    if expected and original['sha256'] is not None and original['sha256'] != expected:
        audit['status'] = 'input_changed'
        return result
    if original['readable']:
        return result
    wanted = normalized_title(title)
    for path in candidate_paths(entry):
        probe = probe_pdf(path)
        title_match = bool(len(wanted) >= 12 and wanted in
                           normalized_title(probe.pop('first_page_text', '')))
        probe['identity_title_match'] = title_match
        audit['attempts'].append(probe)
        if not probe['readable'] or not title_match:
            continue
        audit.update(status='recovered', resolved=probe,
                     cached_parser_discarded=entry.get('parsed_json'))
        result.update(pdf=str(path), pdf_sha256=probe['sha256'], pdf_pages=probe['pages'],
                      input_status='ready', parsed_json=None)
        # Cache was bound to the original bytes; do not mix its page IDs with a new PDF.
        return result
    audit['status'] = 'unrecoverable'
    return result
