"""Repair a figure mistakenly attached to a below-table caption using native cell rules."""
from collections import defaultdict
from statistics import median
import re


def recover(page, crop):
    import fitz
    old = fitz.Rect(crop['bbox'])
    if not crop.get('caption_bbox'):
        return None
    caption = fitz.Rect(crop['caption_bbox'])
    if caption.y1 > old.y0 or not any(old.intersects(fitz.Rect(image['bbox']))
                                    for image in page.get_image_info()):
        return None
    columns = defaultdict(list)
    for drawing in page.get_drawings():
        for item in drawing.get('items', []):
            if item[0] != 'l':
                continue
            start, end = item[1:3]
            top, bottom = sorted((start.y, end.y))
            if abs(start.x - end.x) < .5 and bottom - top >= 5 and bottom < caption.y0:
                if caption.x0 < start.x < caption.x1 + 30:
                    columns[round(start.x)].append((top, bottom))
    spans = []
    for position, segments in columns.items():
        merged = []
        for top, bottom in sorted(segments):
            if merged and top - merged[-1][1] < 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], bottom))
            else:
                merged.append((top, bottom))
        spans.extend((position, top, bottom) for top, bottom in merged
                     if bottom - top >= 35 and 0 < caption.y0 - bottom < 30)
    if len(spans) < 3:
        return None
    top, bottom = median(row[1] for row in spans), median(row[2] for row in spans)
    aligned = [row for row in spans if abs(row[1] - top) < 1 and abs(row[2] - bottom) < 1]
    if len(aligned) < 3:
        return None
    lines = [(fitz.Rect(line['bbox']), ''.join(span['text'] for span in line['spans']).strip())
             for block in page.get_text('dict')['blocks'] for line in block.get('lines', [])]
    cells = [(box, text) for box, text in lines if box.y0 >= top - 2 and box.y0 < bottom and box.y1 < caption.y0 - 1
             and box.x0 >= caption.x0 - 2 and box.x1 <= caption.x1 + 35 and text]
    rows = []
    for box, text in sorted(cells, key=lambda row: (row[0].y0, row[0].x0)):
        if not rows or abs(box.y0 - median(item[0].y0 for item in rows[-1])) > 2.5:
            rows.append([])
        rows[-1].append((box, text))
    if len(rows) < 4 or any(len(row) < 3 for row in rows):
        return None
    if any(sum(bool(re.search(r'\d', text)) for _, text in row) < 2
           or any(len(text) > 70 for _, text in row) for row in rows[1:]):
        return None
    if any(min(box.y0 for box, _ in later) - max(box.y1 for box, _ in earlier) > 15
           for earlier, later in zip(rows, rows[1:])):
        return None
    box = fitz.Rect(min(rect.x0 for rect, _ in cells) - 2, top - 2,
                    max(rect.x1 for rect, _ in cells) + 2, max(bottom, max(rect.y1 for rect, _ in cells)) + .5)
    for drawing in page.get_drawings():
        for item in drawing.get('items', []):
            if item[0] != 'l':
                continue
            start, end = item[1:3]
            if abs(start.y - end.y) < .5 and top <= start.y <= bottom:
                if abs(start.x - end.x) > box.width * .8:
                    box.x0 = min(box.x0, start.x - 2, end.x - 2)
                    box.x1 = max(box.x1, start.x + 2, end.x + 2)
    if not page.rect.contains(box) or box.y1 >= caption.y0:
        return None
    if any(box.intersects(fitz.Rect(image['bbox'])) for image in page.get_image_info()):
        return None
    return list(box), dict(version='column_table_reassociation', original_bbox=list(old),
        bbox=list(box), changed=True, completeness_verified=True,
        status='verified_caption_adjacent_native_cell_columns', data_rows=len(rows) - 1,
        aligned_vertical_rules=len(aligned), no_source_pixels_removed=False,
        replaces_misassociated_raster_figure=True, source_caption_unchanged=True,
        scope='native cell structure and caption adjacency; numerical transcription not certified')
