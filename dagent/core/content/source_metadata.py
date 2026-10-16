"""Conservative, model-free validation of proposed author names in front matter.

Unlike a contiguous full-author-string test, affiliations may separate names.
Never infer a name from an email or match a name inside body prose. Ambiguity
leaves the existing metadata unchanged. Source support is not completeness.
"""
import re
import unicodedata
from pathlib import Path

VERSION = 'source_author_lines'


def _normalize(text):
    accents = {'´': '\u0301', '`': '\u0300', 'ˆ': '\u0302', '¨': '\u0308', '˜': '\u0303'}
    text = re.sub(r'([´`ˆ¨˜])([^\W\d_])',
                  lambda match: match[2] + accents[match[1]], text)
    return unicodedata.normalize('NFKC', text)


def _key(text):
    text = re.sub(r'<sup>.*?</sup>', '', text, flags=re.I)
    text = re.sub(r'<[^>]*>', '', text)
    return tuple(re.findall(r'[^\W\d_]+', _normalize(text).casefold()))


def _names(text):
    # "Surname, Given" is intentionally not inferred; ambiguous input fails closed.
    return [part.strip() for part in re.split(r'\s*(?:[,;\n&]|\band\b)\s*', text) if part.strip()]


def _display_name(text):
    text = re.sub(r'<sup>.*?</sup>', '', text, flags=re.I)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[\d*∗†‡]+', '', _normalize(text))
    return ' '.join(text.split())


def _source_names(line):
    segments = _names(line)
    if len(segments) == 1 and re.search(r'\d', _normalize(line)):
        pieces = [part.strip() for part in re.split(r'[\d\s]*\d[\d\s]*', _normalize(line)) if part.strip()]
        if len(pieces) > 1 and all(len(_key(part)) >= 2 for part in pieces):
            return pieces
    return segments


def _native_rows(matches):
    rows = []
    for match in sorted(matches, key=lambda item: item['bbox'][1]):
        if not rows or match['bbox'][1] - rows[-1][0]['bbox'][1] > 3:
            rows.append([])
        rows[-1].append(match)
    if any(max(item['bbox'][3] for item in upper) >= min(item['bbox'][1] for item in lower)
           for upper, lower in zip(rows, rows[1:])):
        return None
    ordered = []
    for row in rows:
        row.sort(key=lambda item: (item['bbox'][0], item['segment']))
        for left, right in zip(row, row[1:]):
            same_line = left['bbox'] == right['bbox'] and left['source_line'] == right['source_line']
            if not same_line and left['bbox'][2] > right['bbox'][0]:
                return None
        ordered.extend(row)
    return ordered


def validate_authors(document, proposed):
    audit = dict(version=VERSION, status='unverified', changed=False, matches=[],
                 author_list_completeness_verified=False, model_calls=0)
    if not isinstance(proposed, str) or not proposed.strip():
        return {**audit, 'reason': 'missing_proposal'}
    names = _names(proposed)
    if not names or any(len(_key(name)) < 2 for name in names):
        return {**audit, 'reason': 'ambiguous_name_format'}
    keys = [_key(name) for name in names]
    if len(keys) != len(set(keys)):
        return {**audit, 'reason': 'duplicate_author'}
    pages = [b.get('page') for b in document.blocks if isinstance(b.get('page'), int)]
    if not pages:
        return {**audit, 'reason': 'missing_first_page'}
    first = min(pages)
    blocks = [b for b in document.blocks if b.get('page') == first]
    # A strong boundary is required: never scan the first 25 arbitrary body blocks.
    boundaries = [i for i, b in enumerate(blocks) if re.match(
        r'^(?:(?:\d+\.?\s*)?(?:abstract|introduction))\b', b['text'].strip(), re.I)]
    if not boundaries:
        return {**audit, 'reason': 'no_frontmatter_boundary'}
    front = blocks[:min(boundaries)]
    forbidden = re.compile(r'@|\b(?:university|department|institute|laboratory|proceedings|copyright)\b', re.I)
    hits = {key: [] for key in keys}
    for order, block in enumerate(front):
        if _key(block['text']) == _key(document.title):
            continue
        for line_no, line in enumerate(block['text'].splitlines()):
            if forbidden.search(line):
                continue
            for segment_no, segment in enumerate(_source_names(line)):
                key = _key(segment)
                if key in hits:
                    hits[key].append(dict(source_id=block['id'], page=first, line=line_no,
                                          segment=segment_no, source_line=line,
                                          source_name=_display_name(segment),
                                          block_order=order, bbox=block.get('bbox')))
    if any(len(hits[key]) != 1 for key in keys):
        return {**audit, 'reason': 'unsupported_or_ambiguous_source_name'}
    matches = [dict(name=name, **hits[key][0]) for name, key in zip(names, keys)]
    # PDF parsers sometimes merge two side-by-side author columns into one block.
    # When available, recover native LINE geometry rather than trust block order.
    pdf_path = Path(document.pdf_path)
    native_geometry = False
    if pdf_path.is_file():
        from dagent.core.document import sha256
        if sha256(pdf_path) != document.pdf_sha256:
            return {**audit, 'reason': 'source_pdf_hash_mismatch'}
        import fitz
        with fitz.open(pdf_path) as pdf:
            page_index = max(0, first - 1)
            if page_index >= len(pdf):
                return {**audit, 'reason': 'invalid_first_page'}
            lines = [line for block in pdf[page_index].get_text('dict')['blocks']
                     for line in block.get('lines', [])]
            for match in matches:
                candidates = [line for line in lines
                              if _key(''.join(s['text'] for s in line['spans'])) == _key(match['source_line'])]
                if len(candidates) != 1:
                    return {**audit, 'reason': 'ambiguous_native_author_line'}
                match['source_block_bbox'] = match['bbox']
                match['bbox'] = list(candidates[0]['bbox'])
            native_geometry = True
    # Coordinate reading order only when author blocks are demonstrably one row.
    boxes = [m['bbox'] for m in matches]
    if all(isinstance(b, (list, tuple)) and len(b) == 4 for b in boxes):
        one_row = max(b[1] for b in boxes) - min(b[1] for b in boxes) <= 3
        if one_row:
            matches.sort(key=lambda m: (m['bbox'][0], m['line'], m['segment']))
            order_basis = 'single_geometric_row_left_to_right'
        elif native_geometry and max(b[0] for b in boxes) - min(b[0] for b in boxes) <= 3:
            matches.sort(key=lambda m: (m['bbox'][1], m['segment']))
            order_basis = 'single_geometric_column_top_to_bottom'
        elif len({m['source_id'] for m in matches}) == 1 and not native_geometry:
            matches.sort(key=lambda m: (m['line'], m['segment']))
            order_basis = 'within_source_block'
        elif native_geometry and (ordered := _native_rows(matches)) is not None:
            matches = ordered
            order_basis = 'native_disjoint_rows_top_to_bottom_left_to_right'
        else:
            return {**audit, 'reason': 'ambiguous_multiline_author_order'}
    elif len({m['source_id'] for m in matches}) == 1:
        matches.sort(key=lambda m: (m['line'], m['segment']))
        order_basis = 'within_source_block'
    else:
        return {**audit, 'reason': 'missing_geometry_for_author_order'}
    value = ', '.join(m['source_name'] for m in matches)
    audit.update(status='source_supported', changed=value != document.authors,
                 previous_authors=document.authors, authors=value, matches=matches,
                 order_basis=order_basis, native_line_geometry_verified=native_geometry)
    document.authors = value
    return audit


AUTHOR_REQUIREMENT_VERSION = 'required_source_authors'


class AuthorMetadataError(ValueError):
    """Missing/unverified attribution is a generation error, not a style warning."""


def required_author_audit(document):
    """Require source-supported names before selection; never invent a fallback.

    This is a presence/provenance gate, NOT proof that every source author was
    extracted. Keep that distinction explicit even when the gate passes.
    """
    source = document.parsing.get('author_metadata_audit', {})
    authors = document.authors
    errors = []
    if not isinstance(authors, str) or not authors.strip():
        errors.append('missing_author_metadata')
    if source.get('status') != 'source_supported':
        errors.append('author_metadata_not_source_supported')
    matches = source.get('matches', [])
    if not matches or source.get('authors') != authors or authors != ', '.join(
            str(m.get('source_name', '')) for m in matches):
        errors.append('author_metadata_does_not_match_source_audit')
    return dict(version=AUTHOR_REQUIREMENT_VERSION, required=True, valid=not errors,
                errors=errors, authors=authors,
                source_ids=list(dict.fromkeys(m['source_id'] for m in matches if 'source_id' in m)),
                protected_from_content_selection=True, model_calls=0,
                author_list_completeness_verified=source.get('author_list_completeness_verified', False),
                source_validation=source)


def require_authors(document):
    audit = required_author_audit(document)
    if not audit['valid']:
        raise AuthorMetadataError('Required author metadata failed: ' + ', '.join(audit['errors']))
    return audit


def author_preservation_errors(document, layout):
    """Compare final attribution to the document, not two possibly empty copies."""
    errors = list(required_author_audit(document)['errors'])
    elements = [e for e in layout.elements if e.id == 'authors' or e.kind == 'authors']
    if len(elements) != 1:
        return errors + ['missing_or_duplicate_author_element']
    element = elements[0]
    if element.id != 'authors' or element.kind != 'authors' or element.text != document.authors:
        errors.append('author_names_changed_or_omitted')
    if (element.font_pt < 28 or element.width <= 0 or element.height <= 0
            or element.x < 0 or element.y < 0
            or element.x + element.width > layout.width + 1e-6
            or element.y + element.height > layout.height + 1e-6):
        errors.append('author_element_not_readable_on_canvas')
    return errors
