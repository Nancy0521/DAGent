"""Recover a complete ruled table from native source marks, without OCR or LLMs.

V26-only repair: a header crop is not a complete table merely because all glyphs
inside it survive. Require matching full-width top/header/bottom rules and a
continuous multi-column text body. Ambiguous continuations fail closed instead
of silently certifying a partial source table. Unruled/raster tables remain
explicitly unverified, not claimed complete.
"""
from statistics import median


def complete_ruled_table_crop(page, crop):
    import fitz
    import re
    from dagent.core.native_assets import CAPTION
    from dagent.core.layout.table_association import recover
    reassociated = recover(page, crop)
    if reassociated is not None:
        return reassociated
    old = fitz.Rect(crop['bbox'])
    report = dict(version='ruled_table_crop', original_bbox=list(old),
                  changed=False, completeness_verified=False, status='unverified_no_rule_structure')
    if old.is_empty or not page.rect.contains(old):
        raise ValueError('Invalid table source crop')
    if not crop.get('caption_bbox'):
        return list(old), report
    caption = fitz.Rect(crop['caption_bbox'])
    below = caption.y1 <= old.y0
    above = caption.y0 >= old.y1
    if not below and not above:
        raise ValueError('Table crop intersects its caption')
    low, high = (caption.y1+1, page.rect.y1) if below else (page.rect.y0, caption.y0-1)
    blocks = page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']
    lines = [(fitz.Rect(line['bbox']), ''.join(s['text'] for s in line['spans']).strip())
             for block in blocks for line in block.get('lines', [])]
    for block in blocks:
        text = ' '.join(''.join(s['text'] for s in line['spans']) for line in block.get('lines', []))
        rect = fitz.Rect(block['bbox'])
        if not CAPTION.match(text) or rect.intersects(caption):
            continue
        overlap = min(rect.x1, old.x1)-max(rect.x0, old.x0)
        if overlap <= old.width*.2:
            continue
        if rect.y1 <= old.y0:
            low = max(low, rect.y1+1)
        if rect.y0 >= old.y1:
            high = min(high, rect.y0-1)
    rules = []
    for drawing in page.get_drawings():
        for item in drawing.get('items', []):
            if item[0] == 'l':
                a, b = item[1:3]
                if abs(a.y-b.y) > .6:
                    continue
                x0, x1, y = min(a.x, b.x), max(a.x, b.x), (a.y+b.y)/2
                if x1-x0 >= max(45, old.width*.8) and low <= y <= high:
                    if abs(x0-old.x0) <= 24 and abs(x1-old.x1) <= 24:
                        rules.append((x0, x1, y))
    rules = sorted(set(rules), key=lambda v: v[2])
    if len(rules) < 3:
        return list(old), report
    # Anchor on the top rule already present in the selected crop, not unrelated
    # graphics elsewhere in the same caption zone.
    tops = [r for r in rules if old.y0-.6 <= r[2] <= old.y1 and r[2]-low <= 65]
    if not tops:
        from dagent.core.layout.middle_table_header import recover as recover_header
        recovered = recover_header(rules, lines, old)
        if recovered is None:
            return list(old), report
        tops = [recovered]
        report['header_recovered_above_partial_crop'] = True
    top = tops[0]
    tolerance = max(3., (top[1]-top[0])*.02)
    aligned = [r for r in rules if r[2] >= top[2] and
               abs(r[0]-top[0]) <= tolerance and abs(r[1]-top[1]) <= tolerance]
    if len(aligned) < 3:
        return list(old), report

    def body_rows(start, end):
        segments = [(r, text) for r, text in lines if text and start < r.y0 and r.y1 < end
                    and r.x0 >= top[0]-2 and r.x1 <= top[1]+2]
        rows = []
        for rect, text in sorted(segments, key=lambda v: (v[0].y1, v[0].x0)):
            if not rows or abs(rect.y1-median(r.y1 for r, _ in rows[-1])) > 2.5:
                rows.append([])
            rows[-1].append((rect, text))
        return rows

    valid = []
    header = aligned[1]
    for bottom in aligned[2:]:
        rows = body_rows(header[2], bottom[2])
        if len(rows) < 2:
            continue
        # A source-native multi-column body: prose after a table is not a row.
        if any(len(row) < 3 or sum(bool(re.search(r'\d', text)) for _, text in row) < 2
               or any(len(text) > 115 or CAPTION.match(text) for _, text in row) for row in rows):
            continue
        heights = [r.height for row in rows for r, _ in row]
        typical = median(heights)
        gaps = [min(r.y0 for r, _ in rows[0])-header[2]]
        gaps += [min(r.y0 for r, _ in nxt)-max(r.y1 for r, _ in prev)
                 for prev, nxt in zip(rows, rows[1:])]
        gaps += [bottom[2]-max(r.y1 for r, _ in rows[-1])]
        if max(gaps) > max(12, typical*2.5):
            continue
        # Repeated cell starts are structural evidence that we have followed the
        # same table rather than a neighbouring paragraph or separate figure.
        anchors = [r.x0 for r, _ in rows[0]]
        if any(sum(any(abs(r.x0-x) < max(6, typical*1.5) for x in anchors) for r, _ in row) < 3
               for row in rows[1:]):
            continue
        valid.append((bottom, rows))
    if not valid:
        if aligned[-1][2] > old.y1+1:
            raise ValueError('Possible missing table rows; native ruled-table continuation is ambiguous')
        return list(old), report
    # The furthest continuously supported rule includes internal group rules;
    # never pick the first candidate merely because it produces a smaller crop.
    bottom, rows = valid[-1]
    box = fitz.Rect(min(old.x0, top[0]-2), min(old.y0, top[2]-2),
                    max(old.x1, bottom[1]+2), max(old.y1, bottom[2]+2))
    if box.y0 < low or box.y1 > high or not page.rect.contains(box):
        raise ValueError('Full table would cross its caption/page boundary')
    old_rows = sum(all(old.contains(r) for r, _ in row) for row in rows)
    return list(box), dict(report, bbox=list(box), changed=box != old,
        completeness_verified=True, status='verified_continuous_rule_enclosure',
        data_rows=len(rows), original_complete_data_rows=old_rows,
        recovered_data_rows=len(rows)-old_rows, top_rule_y=top[2], header_rule_y=header[2],
        bottom_rule_y=bottom[2], no_source_pixels_removed=True,
        scope='single-page native ruled body; semantic accuracy and multi-page continuation not certified')
