"""Recover partially clipped vector glyphs before identical-source PDF sampling.

Expansion only, never erasing source marks. Bound by original PDF identity,
caption boundaries, small coordinate/area limits, and zero remaining cut glyphs.
Raster-internal labels are not certified by these vector checks.
"""
from copy import deepcopy
from pathlib import Path
from dagent.core.config import dump_json
from dagent.core.document import sha256
from dagent.core.layout.source_visuals import prepare_visuals


def complete_crop(page, crop, *, max_expansion_pt=16., max_area_ratio=1.20):
    import fitz
    old=fitz.Rect(crop['bbox']);box=fitz.Rect(old)
    caption=fitz.Rect(crop['caption_bbox']) if crop.get('caption_bbox') else None
    if old.is_empty or not page.rect.contains(old):
        raise ValueError('Invalid original source rectangle')
    allowed=old+(-max_expansion_pt,-max_expansion_pt,max_expansion_pt,max_expansion_pt)
    allowed &= page.rect
    if caption:
        if caption.y0>=old.y1:
            allowed.y1=min(allowed.y1,caption.y0-1.)
        elif caption.y1<=old.y0:
            allowed.y0=max(allowed.y0,caption.y1+1.)
        else:
            raise ValueError('Original figure already intersects its own main caption')
    chars=[]
    for block in page.get_text('rawdict',flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']:
        for line in block.get('lines',[]):
            for span in line.get('spans',[]):
                for char in span.get('chars',[]):
                    if char['c'].strip():
                        chars.append(fitz.Rect(char['bbox']))
    def partial(b):
        return [c for c in chars if b.intersects(c) and not b.contains(c)]
    original_count=len(partial(old))
    for _ in range(6):
        cut=partial(box)
        if not cut:
            break
        proposed=fitz.Rect(box)
        for c in cut:
            proposed |= c+(-.25,-.25,.25,.25)
        if not allowed.contains(proposed) or proposed.get_area()>old.get_area()*max_area_ratio:
            raise ValueError('Completing cut source labels would cross the safe caption/area boundary')
        box=proposed
    remaining=len(partial(box))
    if remaining:
        raise ValueError('Source crop still intersects partial vector glyphs')
    return list(box),dict(original_bbox=list(old),repaired_bbox=list(box),
                         original_partial_vector_glyphs=original_count,remaining_partial_vector_glyphs=remaining,
                         expanded=box!=old,area_ratio=box.get_area()/old.get_area(),
                         operation='bounded_expansion_to_complete_intersecting_source_glyphs',
                         raster_embedded_text_assessed=False)


def prepare_quality_visuals(tree,document,output,*,target_width_inches,minimum_dpi):
    import fitz
    doc=deepcopy(document);changes=[]
    paths={str(Path(n['asset_path']).resolve()) for s in tree['children'] for n in s['children'] if n.get('asset_path')}
    if sha256(Path(doc['pdf_path'])) != doc['pdf_sha256']:
        raise ValueError('Source PDF identity mismatch')
    selected=[]
    with fitz.open(doc['pdf_path']) as pdf:
        for a in doc['assets']:
            if str(Path(a['path']).resolve()) not in paths:
                continue
            crop=a.get('source_crop')
            if not crop or crop.get('pdf_sha256')!=doc['pdf_sha256']:
                raise ValueError('Quality pilot requires verified PDF crop provenance')
            if type(crop.get('page')) is not int or not 1<=crop['page']<=len(pdf):
                raise ValueError('Invalid source page')
            new,change=complete_crop(pdf[crop['page']-1],crop)
            a['source_crop']['bbox']=new
            changes.append(dict(asset_id=a['id'],page=crop['page'],pdf_sha256=doc['pdf_sha256'],**change))
            selected.append((a['id'],crop['page'],fitz.Rect(new)))
    if len(selected)!=len(paths):
        raise ValueError('Selected source visual missing from document')
    for i,(aid,page,bbox) in enumerate(selected):
        for bid,other_page,other in selected[i+1:]:
            if page==other_page and (bbox & other).get_area()/min(bbox.get_area(),other.get_area())>.10:
                raise ValueError('Selected figure crops overlap, risking duplicate visual evidence: '+aid+'/'+bid)
    sampled,audit=prepare_visuals(tree,doc,output,target_width_inches=target_width_inches,minimum_dpi=minimum_dpi)
    for row in audit:
        row['crop_recovery']=next(c for c in changes if c['asset_id']==row['asset_id'])
        if row.get('partially_clipped_source_glyphs'):
            raise ValueError('Resampled crop still has clipped vector text')
    dump_json(Path(output)/'audit.json',audit)
    dump_json(Path(output)/'crop_recovery.json',changes)
    return sampled,audit
