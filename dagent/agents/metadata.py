"""Hash-verified, model-free author recovery from native PDF front matter.

Names are copied, never invented or fuzzy-matched. Parsed blocks may flatten
names/emails/affiliations, so native lines are read BEFORE rejecting a proposal.
Presence/provenance is verified; completeness and correct authorship still need
human review. No benchmark answers or previous generated posters are inputs.
"""
from copy import deepcopy
from pathlib import Path
import re
import unicodedata

from dagent.core.config import dump_json
from dagent.core.document import sha256
from dagent.core.content.source_metadata import required_author_audit

VERSION = 'dagent_native_authors'
MARKS = r'[\d*∗†‡§¶‖⋆♠♣♥♦✉]+'
PARTICLES = {'de', 'del', 'da', 'dos', 'van', 'von', 'der', 'den', 'di', 'du', 'le', 'la', 'al', 'el', 'y', 'della'}
NON_NAMES = re.compile(r'\b(?:universit\w*|department\w*|institute|institut|laborator\w*|laboratoire|'
    r'faculty|school|college|research|centre|center|division|unit|engineering|technology|technische|'
    r'labs?|group|inc|corp|corporation|company|computers?|vision|artificial|intelligence|systems|'
    r'aws|ai|uc|eth|amc|unidistance|suisse|zurich|diego|barbara|merced|'
    r'google|microsoft|amazon|nvidia|sony|siemens|adobe|deepmind|independent|editor|abstract|'
    r'introduction|preprint|copyright|published|submitted|revised|proceedings|anonymous|authors?|'
    r'equal|corresponding|contributed|contribution|foundation|science|sciences|computational|'
    r'physics|mathematics|informatique|polytechnique|berkeley|cambridge|munich|london|beijing|'
    r'china|germany|switzerland|france|japan|kingdom|netherlands|january|february|march|april|'
    r'may|june|july|august|september|october|november|december)\b', re.I)


def normalize(text):
    accents = {'´': '\u0301', '`': '\u0300', 'ˆ': '\u0302', '¨': '\u0308', '˜': '\u0303'}
    text = re.sub(r'([´`ˆ¨˜])([^\W\d_])', lambda m: m[2] + accents[m[1]], text)
    return unicodedata.normalize('NFKC', text).replace('\u00ad', '')


def key(text):
    return ''.join(c for c in normalize(text).casefold() if c.isalpha())


def clean(text):
    text = re.sub(MARKS, '', normalize(text))
    return ' '.join(text.strip(' ,;&\\').split())


def is_name(text):
    if not text or NON_NAMES.search(text) or re.search(r'[@:{}()/=<>_]', text):
        return False
    if text.isupper() and any(len(w.strip('.')) > 2 for w in text.split()):
        return False  # title/organization smallcaps, not an unambiguous name
    words = text.split()
    if not 2 <= len(words) <= 7:
        return False
    for word in words:
        if word.casefold().strip('.') in PARTICLES:
            continue
        parts = re.split(r"[-’']", word)
        if not word[0].isupper() or any(not p or not all(c.isalpha() or c == '.' for c in p) for p in parts):
            return False
    return True


def split_names(text):
    # Numeric superscripts delimit ICML's space-separated author lists.
    text = normalize(text)
    text = re.sub(r'(?<=\w)[*∗†‡§¶‖⋆]+(?=[A-Z])', ',', text)
    text = re.sub(r'\s*\d+(?:\s*[,\s]\s*\d+)*\s*', ',', text)
    return [clean(p) for p in re.split(r'\s*(?:[,;&]|\band\b)\s*', text) if clean(p)]


def native_frontmatter(document):
    import fitz
    pdf_path = Path(document.pdf_path)
    if not pdf_path.is_file() or sha256(pdf_path) != document.pdf_sha256:
        raise ValueError('source_pdf_hash_mismatch')
    with fitz.open(pdf_path) as pdf:
        if not len(pdf):
            raise ValueError('source_pdf_has_no_pages')
        page = pdf[0]
        lines = []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                if abs(line.get('dir', (1, 0))[1]) > .1:
                    continue  # rotated arXiv stamps are not front-matter names
                spans = line.get('spans', [])
                text = ''.join(s['text'] for s in spans).strip()
                if not text:
                    continue
                significant = [s for s in spans if key(s['text'])] or spans
                lines.append(dict(text=text, bbox=list(line['bbox']), page=1,
                    source_id=f'native_p1_l{len(lines)}',
                    font_size=max(s['size'] for s in significant),
                    font=significant[0]['font'],
                    spans=[dict(text=s['text'],bbox=list(s['bbox']),size=s['size'],font=s['font']) for s in spans]))
        # Require a title/body boundary, not an arbitrary first N text blocks.
        boundaries = [l['bbox'][1] for l in lines if re.match(
            r'^(?:abstract|summary|(?:1[.\s]+)?introduction)\b', l['text'].strip(), re.I)
            and l['bbox'][1] > 60]
        caption_bounds = [l['bbox'][1] for l in lines if re.match(r'^(?:figure|fig\.)\s*1[.:\s]',l['text'],re.I)]
        upper = min(boundaries + caption_bounds, default=page.rect.height * .80)
        eligible = [l for l in lines if 40 < l['bbox'][1] < upper and l['bbox'][0] > page.rect.width * .04
                    and not re.search(r'arxiv|https?://|doi:|@', l['text'], re.I)]
        if not eligible:
            raise ValueError('native_frontmatter_not_found')
        title_key = key(document.title)
        largest = max(l['font_size'] for l in eligible)
        title_lines = [l for l in eligible if len(key(l['text'])) >= 12 and key(l['text']) in title_key
                       and l['font_size'] >= largest * .76]
        if not title_lines:
            title_lines = [l for l in eligible if l['font_size'] >= largest * .96]
            if largest < 12:
                raise ValueError('native_title_boundary_ambiguous')
        # The cached title may itself be truncated. Include native continuation
        # lines of the same large title style; never treat those as names.
        title_size = max(l['font_size'] for l in title_lines)
        title_lines += [l for l in eligible if l['font_size'] >= title_size * .96]
        title_bottom = max(l['bbox'][3] for l in title_lines)
        front = [l for l in lines if l['bbox'][1] >= title_bottom - 1 and l['bbox'][1] < upper]
        return front, dict(page=1, title_bottom=title_bottom, body_top=upper,
                           strong_body_boundary=bool(boundaries or caption_bounds),pdf_sha256=document.pdf_sha256)


def recover_authors(document, output=None):
    """Up to three local strategies, with immutable attempt-specific feedback."""
    lines, region = native_frontmatter(document)
    history = []
    # A pass does not mean we know that every author was recovered.
    for attempt in range(3):
        matches, rejected = [], []
        candidates = []
        pending = None
        for line in sorted(lines, key=lambda l: (round(l['bbox'][1] / 3), l['bbox'][0])):
            text = line['text']
            if '@' in text or NON_NAMES.search(text) or line['font_size'] < 8:
                rejected.append(dict(source_id=line['source_id'],reason='not_author_line'))
                pending = None
                continue
            parts = split_names(text)
            if attempt >= 1 and pending is not None:
                previous, fragment = pending
                if (parts and line['bbox'][1] - previous['bbox'][3] < 20
                        and abs(line['bbox'][0] - previous['bbox'][0]) < 5
                        and is_name(fragment + ' ' + parts[0])):
                    candidates.append((fragment+' '+parts.pop(0), [previous,line]))
                pending = None
            for i, part in enumerate(parts):
                if is_name(part):
                    candidates.append((part,[line]))
                elif attempt >= 1 and i == len(parts)-1 and len(part.split()) == 1 and part[:1].isupper():
                    pending = (line,part)
                else:
                    rejected.append(dict(source_id=line['source_id'],reason='not_person_name',text=part))
        mixed_style = False
        if candidates:
            anchor = candidates[0][1][0]
            # Names share a style. This excludes affiliations/figure labels even
            # when their wording happens to look like a capitalized person name.
            for name, sources in candidates:
                line = sources[0]
                same_font = line['font'] == anchor['font']
                same_size = abs(line['font_size']-anchor['font_size']) < .65
                if not same_size:
                    continue
                if attempt < 2 and not same_font:
                    mixed_style = True
                    continue
                if key(name) in {key(m['source_name']) for m in matches}:
                    continue
                matches.append(dict(source_id=sources[0]['source_id'],page=1,
                    source_name=name,source_line=' '.join(s['text'] for s in sources),
                    bbox=sources[0]['bbox'],source_lines=deepcopy(sources),
                    pdf_sha256=document.pdf_sha256))
        errors = [] if matches else ['no_unambiguous_native_author_names']
        # If line wrapping leaves a dangling capitalized name fragment, require
        # the next strategy rather than silently advertising a complete list.
        fragments = [r for r in rejected if r.get('reason')=='not_person_name' and
                     len(r.get('text','').split()) == 1 and r.get('text','')[:1].isupper()]
        if mixed_style:
            errors.append('mixed_author_fonts_use_same_size_recovery')
        if attempt == 0 and fragments:
            errors.append('possible_wrapped_name_use_line_join_recovery')
        record = dict(attempt=attempt+1,strategy=('native_lines_same_style','join_wrapped_native_names','native_lines_same_size')[attempt],
            feedback=history[-1]['errors'] if history else [],errors=errors,
            recovered_count=len(matches),matches=matches,rejected=rejected)
        history.append(record)
        if output:
            dump_json(Path(output)/'metadata_attempts'/f'attempt_{attempt+1}.json',record)
        if not errors:
            value=', '.join(m['source_name'] for m in matches)
            audit=dict(version=VERSION,status='source_supported',authors=value,matches=matches,
                previous_authors=document.authors,changed=value != document.authors,
                author_list_completeness_verified=False,native_line_geometry_verified=True,
                order_basis='native_rows_top_to_bottom_left_to_right',region=region,
                model_calls=0,attempts_used=attempt+1,attempt_limit=3)
            document.authors=value
            document.parsing['author_metadata_audit']=audit
            document.parsing['author_recovery_history']=history
            if output:
                dump_json(Path(output)/'author_metadata_audit.json',audit)
                dump_json(Path(output)/'required_author_metadata.json',required_author_audit(document))
            return audit
    audit=dict(version=VERSION,status='unverified',matches=[],reason='native_author_recovery_exhausted',
               attempts=history,region=region,model_calls=0)
    document.parsing['author_metadata_audit']=audit
    if output:dump_json(Path(output)/'author_metadata_audit.json',audit)
    raise ValueError('native_author_recovery_exhausted')
