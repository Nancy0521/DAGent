"""Glyph-level checks for the exported poster PDF."""
from __future__ import annotations

import unicodedata

def normalized_text(text):
    # Whitespace and typographic ligatures only: keep scientific symbols distinct.
    table = str.maketrans({'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl'})
    return ''.join(text.translate(table).split())


def spatial_text(chars, baseline_tolerance):
    """Restore reading order when fallback-font runs are emitted after a line."""
    # A combining mark can be emitted in a fallback-font span after the
    # ordinary text. Its zero-advance origin is the *end* of its base glyph,
    # sometimes slightly after the following character due to export rounding.
    # Sort the whole grapheme at its base's origin, not at the mark's origin.
    # This never removes a mark or changes its identity/position on the page.
    units = []
    for char in chars:
        if all(unicodedata.combining(c) for c in char['c']):
            candidates = [base for base in chars
                if not any(unicodedata.combining(c) for c in base['c'])
                and abs(base['origin'][1] - char['origin'][1]) <= .1
                and base['origin'][0] < char['origin'][0]
                and abs(base['bbox'][2] - char['origin'][0]) <= .1]
            if len(candidates) == 1:
                units.append((candidates[0]['origin'][0], 1, char))
                continue
        units.append((char['origin'][0], 0, char))
    rows = []
    for x, order, char in sorted(units, key=lambda item: (item[2]['origin'][1], item[0], item[1])):
        baseline = char['origin'][1]
        if not rows or baseline - rows[-1][0] > baseline_tolerance:
            rows.append((baseline, []))
        rows[-1][1].append((x, order, char))
    return '\n'.join(''.join(c['c'] for _, _, c in sorted(row, key=lambda item: (item[0], item[1])))
                     for _, row in rows)


def inspect_rendered_text(page, layout, tolerance=.75):
    """Check actual glyphs per textbox, not substrings found elsewhere on the page.

    Each glyph is assigned once by baseline position. Full sequence comparison
    detects missing short/repeated text; glyph bounds detect box/canvas overflow.
    A small point tolerance covers export rounding, not lost lines.
    """
    import fitz
    elements = [e for e in layout.elements if e.kind != 'figure' and e.text]
    boxes = [fitz.Rect(e.x*72, e.y*72, (e.x+e.width)*72, (e.y+e.height)*72)
             for e in elements]
    assigned = [[] for _ in elements]
    defects, reported = [], set()

    def contains(box, other):
        return fitz.Rect(box.x0-tolerance, box.y0-tolerance,
                         box.x1+tolerance, box.y1+tolerance).contains(other)

    def report(category, target, description):
        key = (category, target)
        if key not in reported:
            defects.append(dict(category=category, severity='hard', target=target,
                                description=description[:160]))
            reported.add(key)

    # Excluding image data avoids loading large raster figures into rawdict.
    raw = page.get_text('rawdict', flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES)
    for block in raw['blocks']:
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                for char in span.get('chars', []):
                    if not normalized_text(char['c']):
                        continue
                    rect, origin = fitz.Rect(char['bbox']), fitz.Point(char['origin'])
                    if not contains(page.rect, rect):
                        report('rendered_clipping', 'canvas', f"Glyph outside canvas: {char['c']}")
                    owners = [i for i, box in enumerate(boxes) if contains(box, origin)]
                    if not owners:
                        # No textbox owns this glyph. Do not silently ignore overflow.
                        nearest = min(range(len(boxes)), key=lambda i:
                            max(boxes[i].x0-origin.x, 0, origin.x-boxes[i].x1)**2 +
                            max(boxes[i].y0-origin.y, 0, origin.y-boxes[i].y1)**2) if boxes else None
                        target = elements[nearest].id if nearest is not None else 'canvas'
                        report('rendered_textbox_overflow', target, f"Glyph outside all textboxes: {char['c']}")
                        continue
                    owner = min(owners, key=lambda i:
                                (origin.y-boxes[i].y0)**2 + (origin.x-boxes[i].x0)**2)
                    assigned[owner].append(char)
                    if not contains(boxes[owner], rect):
                        report('rendered_textbox_overflow', elements[owner].id,
                               f"Glyph extends beyond textbox: {char['c']}")
    for element, chars in zip(elements, assigned):
        actual = normalized_text(spatial_text(chars, max(1., element.font_pt*.08)))
        expected = normalized_text(element.text)
        if expected != actual:
            report('rendered_missing_text', element.id,
                   f"Textbox text differs (expected {len(expected)}, rendered {len(actual)} chars): {element.text[:80]}")
    return defects

