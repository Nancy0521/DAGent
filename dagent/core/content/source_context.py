"""Budgeted whole-block document context with explicit source visibility audits."""
import hashlib
import json

VERSION = 'whole_blocks'


def sample(document, config, byte_limit=None):
    by_page = {}
    for block in document.blocks:
        by_page.setdefault(block['page'], []).append(block)
    selected, used = [], 0
    capacity = min(config.max_document_chars,
                   byte_limit if byte_limit is not None else config.max_document_chars)
    for j in range(max((len(group) for group in by_page.values()), default=0)):
        for group in by_page.values():
            if j >= len(group):
                continue
            block = group[j]
            item = dict(id=block['id'], page=block['page'], text=block['text'])
            if block.get('kind') in ('table_row', 'table_footnote'):
                item.update(kind=block['kind'], asset_id=block['asset_id'])
            size = len(json.dumps(item, ensure_ascii=False).encode()) + 2
            if used + size <= capacity:
                selected.append(item)
                used += size
    return selected


def audit(document, payload, *, policy, byte_capacity=None):
    if policy != VERSION:
        raise ValueError('Unknown understanding source policy')
    originals = {b['id']: b for b in document.blocks}
    if len(originals) != len(document.blocks):
        raise ValueError('Duplicate original block IDs')
    visible = {}
    for b in payload['blocks']:
        if b['id'] in visible or b['id'] not in originals:
            raise ValueError('Unknown or duplicate understanding block')
        if (not originals[b['id']]['text'].startswith(b['text'])
                or originals[b['id']]['page'] != b['page']):
            raise ValueError('Understanding text/page drifted from source')
        visible[b['id']] = b
    rows = []
    for sid, block in originals.items():
        text = visible.get(sid, {}).get('text', '')
        state = ('omitted' if sid not in visible else
                 'complete' if text == block['text'] else 'truncated')
        rows.append(dict(source_id=sid, page=block['page'], status=state,
                         source_characters=len(block['text']), visible_characters=len(text),
                         source_sha256=hashlib.sha256(block['text'].encode()).hexdigest(),
                         visible_sha256=hashlib.sha256(text.encode()).hexdigest() if sid in visible else None))
    counts = {name: sum(r['status'] == name for r in rows) for name in ('complete', 'truncated', 'omitted')}
    if policy == VERSION and counts['truncated']:
        raise ValueError('Whole-block policy produced a truncated source')
    return dict(version='evidence_context_visibility', policy=policy, blocks=rows,
                counts=counts, total_blocks=len(rows), byte_capacity=byte_capacity,
                serialized_blocks_bytes=sum(len(json.dumps(b, ensure_ascii=False).encode()) + 2
                                            for b in payload['blocks']),
                all_parsed_blocks_visible=counts['complete'] == len(rows),
                semantic_completeness_verified=False, pdf_extraction_completeness_verified=False,
                omitted_blocks_reason='unchanged_context_capacity_or_not_in_payload', model_calls=0)
