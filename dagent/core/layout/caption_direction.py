"""Reassociate existing table assets without changing parser IDs or source captions."""
from copy import deepcopy


def recover(page, crop):
    import fitz
    from dagent.core.native_assets import caption_graphics_crops
    if not crop.get('caption_bbox'):
        return None
    caption = fitz.Rect(crop['caption_bbox'])
    old = fitz.Rect(crop['bbox'])
    matches = [row for row in caption_graphics_crops(page, nearest_table=True)
        if row['kind'] == 'table' and all(abs(before - after) < .1
            for before, after in zip(row['caption_bbox'], caption))]
    if len(matches) != 1:
        return None
    match = matches[0]
    above = old.y1 < caption.y0
    below = old.y0 > caption.y1
    if (above and match['direction'] == 'above') or (below and match['direction'] == 'below'):
        return None
    if not above and not below:
        raise ValueError('Table crop overlaps its own source caption')
    return deepcopy(match['bbox']), dict(version='caption_direction',
        original_bbox=list(old), reassociated_bbox=deepcopy(match['bbox']),
        changed=True, direction=match['direction'], source_caption_unchanged=True,
        source_native_geometry_only=True, semantic_match_certified=False,
        retains_existing_asset_identity=True)
