"""Complete native ruled tables and remove boundary-crossing exterior prose.

A long paragraph cut by the old crop and wholly outside every contained graphic
is not a chart label. Move the boundary past that paragraph, keep all graphics
and all other source glyphs, then perform the existing bounded glyph completion.
Ambiguous text is retained. Ruled-table continuation has a separate structural
audit; ambiguous continuations fail rather than certifying a header-only table.
No inpainting, redrawing, OCR, or manual coordinates.
"""
from copy import deepcopy
from pathlib import Path
import hashlib

from dagent.core.config import dump_json
from dagent.core.document import sha256
from dagent.core.layout.quality_assets import prepare_quality_visuals

VERSION = 'caption_direction_and_ruled_table_guard'


def guarded_crop(page, crop):
    import fitz
    old = fitz.Rect(crop['bbox'])
    if old.is_empty or not page.rect.contains(old):
        raise ValueError('Invalid original crop')
    graphics = [fitz.Rect(d['rect']) for d in page.get_drawings() if d.get('rect') is not None]
    graphics += [fitz.Rect(d['bbox']) for d in page.get_image_info()]
    graphics = [r for r in graphics if old.contains(r) and max(r.width, r.height) >= 2]
    report = dict(version=VERSION, original_bbox=list(old), excluded_prose=[],
                  ambiguous_text_retained=True, graphic_primitives_preserved=True)
    if not graphics:
        return list(old), dict(report, bbox=list(old), changed=False)
    top, bottom = min(r.y0 for r in graphics), max(r.y1 for r in graphics)
    blocks = page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']
    candidates = []
    for block in blocks:
        lines = block.get('lines', [])
        box = fitz.Rect(block['bbox'])
        if not old.intersects(box) or old.contains(box) or len(lines) < 3:
            continue
        texts = [''.join(span['text'] for span in line['spans']) for line in lines]
        # Require multiple prose-like lines, not just a long figure title.
        if sum(len(t.strip()) >= 65 for t in texts) < 3 or box.width < old.width * .75:
            continue
        if box.y0 < old.y0 < box.y1 < top - 1:
            side = 'top'
        elif bottom + 1 < box.y0 < old.y1 < box.y1:
            side = 'bottom'
        else:
            continue
        candidates.append((side, box, lines, texts))
    new = fitz.Rect(old)
    excluded_lines = set()
    for side, box, lines, texts in candidates:
        if side == 'top':
            new.y0 = max(new.y0, box.y1 + .5)
        else:
            new.y1 = min(new.y1, box.y0 - .5)
        excluded_lines.update(tuple(line['bbox']) for line in lines)
        report['excluded_prose'].append(dict(side=side, bbox=list(box), line_count=len(lines),
            text_sha256=hashlib.sha256('\n'.join(texts).encode()).hexdigest()))
    # If ANY other previously intersected text/graphic would be dropped, retain
    # the original crop. This is a structural guard, not a semantic classifier.
    if not all(new.contains(r) for r in graphics):
        return list(old), dict(report, bbox=list(old), changed=False, reason='graphic_conflict')
    for block in blocks:
        for line in block.get('lines', []):
            rect = fitz.Rect(line['bbox'])
            if tuple(line['bbox']) not in excluded_lines and old.intersects(rect):
                if (old & rect).get_area() > (new & rect).get_area() + .01:
                    return list(old), dict(report, bbox=list(old), changed=False, reason='other_text_conflict')
    return list(new), dict(report, bbox=list(new), changed=new != old)


def prepare_guarded_visuals(tree, document, output, *, target_width_inches, minimum_dpi):
    import fitz
    doc = deepcopy(document)
    if sha256(Path(doc['pdf_path'])) != doc['pdf_sha256']:
        raise ValueError('Source PDF identity mismatch')
    selected = {str(Path(n['asset_path']).resolve()) for s in tree['children']
                for n in s['children'] if n.get('asset_path')}
    changes, table_changes = [], []
    with fitz.open(doc['pdf_path']) as pdf:
        for asset in doc['assets']:
            if str(Path(asset['path']).resolve()) not in selected:
                continue
            crop = asset.get('source_crop')
            if not crop or crop.get('pdf_sha256') != doc['pdf_sha256']:
                raise ValueError('Verified source crop required')
            if type(crop.get('page')) is not int or not 1 <= crop['page'] <= len(pdf):
                raise ValueError('Invalid source page')
            if asset.get('kind') == 'table':
                from dagent.core.layout.caption_direction import recover as recover_direction
                direction_change = recover_direction(pdf[crop['page'] - 1], crop)
                if direction_change is not None:
                    crop['bbox'], direction_audit = direction_change
                from dagent.core.layout.table_crop import complete_ruled_table_crop
                box, table_change = complete_ruled_table_crop(pdf[crop['page'] - 1], crop)
                if direction_change is not None:
                    table_change['caption_direction_reassociation'] = direction_audit
                crop['bbox'] = box
                table_changes.append(dict(asset_id=asset['id'], page=crop['page'], **table_change))
            from dagent.core.layout.lateral_prose import recover
            crop['bbox'], lateral_change = recover(pdf[crop['page'] - 1], crop)
            box, change = guarded_crop(pdf[crop['page'] - 1], crop)
            if lateral_change:
                change['lateral_prose_recovery'] = lateral_change
            crop['bbox'] = box
            changes.append(dict(asset_id=asset['id'], page=crop['page'], **change))
    sampled, audit = prepare_quality_visuals(tree, doc, output,
        target_width_inches=target_width_inches, minimum_dpi=minimum_dpi)
    for row in audit:
        row['boundary_prose_guard'] = next(c for c in changes if c['asset_id'] == row['asset_id'])
        row['table_completeness'] = next((c for c in table_changes if c['asset_id'] == row['asset_id']), None)
    dump_json(Path(output) / 'audit.json', audit)
    dump_json(Path(output) / 'boundary_prose_guard.json', changes)
    dump_json(Path(output) / 'table_completeness.json', table_changes)
    return sampled, audit
