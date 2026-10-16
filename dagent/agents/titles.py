"""Deterministic title selection; manifest metadata never becomes body evidence."""
import re
import unicodedata


def _key(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text).casefold()
                   if c.isalnum())


def _reject(text):
    if len(text) < 6 or len(text) > 350 or len(text.split()) > 50:
        return 'implausible_title_length'
    if re.search(r'(?i)^(published as|under review|proceedings of|arxiv\b|doi\b|'
                 r'10\.\d{4,9}/|https?://|copyright|©|all rights reserved|preprint|accepted at)', text):
        return 'publication_header'
    if re.search(r'(?i)\.(?:graffle|pdf|docx?|pptx?|tex|eps|svg|png|jpe?g)$', text):
        return 'source_filename_not_title'
    if re.fullmatch(r'(?i)(abstract|references|introduction|untitled|main(?:\.pdf)?|'
                    r'paper(?:\.pdf)?|manuscript|microsoft word.*|[\d. /-]+)', text):
        return 'generic_metadata_or_heading'
    if '@' in text or re.search(r'(?i)\b(university|department|institute of|laboratory)\b', text):
        return 'likely_affiliation'
    return None


def select_title(entry, metadata_title, blocks, native_blocks=()):
    expected = next((line.split('=', 1)[1].strip()
                     for line in (entry.get('source_metadata') or '').splitlines()
                     if line.startswith('title=')), '')
    metadata_title = ' '.join((metadata_title or '').split())
    candidates, rejected, front_text = [], [], []
    for origin, source in (('parsed', blocks), ('native', native_blocks)):
        if not source:
            continue
        first_page = min(b['page'] for b in source)
        for block in [b for b in source if b['page'] == first_page][:25]:
            text = ' '.join(block['text'].split())
            if re.match(r'(?i)^abstract\b', text):
                break
            front_text.append(text)
            row = dict(text=text, source_id=block['id'], origin=origin)
            reason = _reject(text)
            if reason:
                rejected.append(dict(row, reason=reason))
            else:
                candidates.append(row)
    front_key = _key(' '.join(front_text))
    metadata_source_matched = bool(metadata_title and _key(metadata_title) in front_key)
    selected, source = None, None
    if expected:
        reason = _reject(expected)
        # Permit line/block splits and typographic accent differences, not an
        # unrelated title from a mismatched manifest or a benchmark poster.
        if not reason and _key(expected) in front_key:
            selected, source = ' '.join(expected.split()), 'expected_match'
        else:
            rejected.append(dict(text=expected, origin='manifest',
                                 reason=reason or 'expected_not_source_matched'))
    if selected is None and metadata_title:
        reason = _reject(metadata_title)
        # A PDF property can be a localized application default (for example
        # "PowerPoint 프레젠테이션"), or a stale title from a different project.
        # Do not whitelist languages or shorten the poster title to pass glyph
        # inspection: require support in the actual source front matter.
        if not reason and not metadata_source_matched:
            reason = 'metadata_not_frontmatter_matched'
        if not reason:
            selected, source = metadata_title, 'metadata'
        else:
            rejected.append(dict(text=metadata_title, origin='metadata', reason=reason))
    if selected is None and candidates:
        selected, source = candidates[0]['text'], 'frontmatter_block'
    if selected is None:
        # Explicitly unverified fallback, never reuse a known publication header.
        selected, source = entry['paper_id'].replace('_', ' '), 'fallback'
    return selected, dict(expected_title=expected, metadata_title=metadata_title,
                          metadata_source_matched=metadata_source_matched,
                          selected_title=selected, selected_source=source,
                          candidates=candidates, rejected=rejected,
                          body_blocks_modified=False, model_calls=0)
