"""Caption-zoned PDF graphics crops with deterministic source provenance.

Only real raster/vector graphics can seed a crop. Adjacent captions are hard
boundaries; horizontal/vertical table rules are retained. No OCR/model or
benchmark poster participates in this parser. Ambiguous/missing crops fail
closed when recropping existing assets rather than silently changing IDs.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from dagent.core.config import writable_path

CAPTION = re.compile(r'^\s*(?:Figure|Fig\.|Table)\s*\d+', re.I)

# Some PDF producers encode a single plotted visual as tens of thousands of
# tiny vector fragments.  Keeping every fragment is unnecessary for caption
# zoning and makes the vertical grouping step pathological.  The threshold is
# deliberately high so ordinary vector PDFs retain the existing crop behavior.
DRAWING_INVENTORY_LIMIT = 10000
PATHOLOGICAL_MIN_DRAWING_EDGE = 5


def _bounds(rects):
    import fitz
    # fitz Rect union ignores degenerate rectangles (e.g. horizontal rules).
    return fitz.Rect(min(r.x0 for r in rects), min(r.y0 for r in rects),
                     max(r.x1 for r in rects), max(r.y1 for r in rects))


def _complete_boundary_lines(clip, lines, *, low, high, page_rect):
    """One bounded expansion for short source lines actually cut by a crop.

    This is not semantic figure detection. Never chase disconnected lines,
    iterate into prose, cross caption zones, or change the source pixels. A
    long/ambiguous line stays unresolved rather than expanding a page crop.
    """
    import fitz
    nearby = fitz.Rect(clip.x0-24, clip.y0-24, clip.x1+24, clip.y1+24)
    additions = [rect for rect, text in lines
                 if clip.intersects(rect) and not clip.contains(rect)
                 and 0 < len(text.strip()) <= 115 and not CAPTION.match(text)
                 and nearby.contains(rect) and page_rect.contains(rect)
                 and rect.y0 >= low and rect.y1 <= high]
    if not additions:
        return clip, 0
    completed = _bounds([clip]+additions)
    completed = fitz.Rect(max(page_rect.x0, completed.x0-2), max(low, completed.y0-2),
                          min(page_rect.x1, completed.x1+2), min(high, completed.y1+2))
    return completed, len(additions)


def caption_graphics_crops(page, raw=None, *, nearest_table=False):
    import fitz
    if raw is None:
        raw = [b for b in page.get_text('blocks',sort=True) if b[6]==0 and b[4].strip()]
    captions = [b for b in raw if CAPTION.match(b[4])]
    drawing_rects = [fitz.Rect(d['rect']) for d in page.get_drawings() if d.get('rect') is not None]
    if len(drawing_rects) > DRAWING_INVENTORY_LIMIT:
        # A pathological vector inventory is dominated by tiny path fragments.
        # Retain real-sized boundaries and long rules, which are the only
        # geometry needed by the caption-zoned crop policy.
        drawing_rects = [r for r in drawing_rects
                         if max(r.width, r.height) >= PATHOLOGICAL_MIN_DRAWING_EDGE]
    graphics = drawing_rects
    graphics += [fitz.Rect(d['bbox']) for d in page.get_image_info()]
    lines = [(fitz.Rect(line['bbox']), ''.join(s['text'] for s in line['spans']))
             for block in page.get_text('dict')['blocks'] if block.get('type')==0
             for line in block.get('lines',[])]
    result = []
    for caption in captions:
        cr = fitz.Rect(caption[:4])
        is_table = bool(re.match(r'^\s*Table',caption[4],re.I))
        aligned = [fitz.Rect(b[:4]) for b in captions if b is not caption
                   and min(b[2],cr.x1)-max(b[0],cr.x0)>min(b[2]-b[0],cr.width)*.2]
        top = max([max(page.rect.y0,cr.y0-page.rect.height*.40)] + [r.y1+1 for r in aligned if r.y1<=cr.y0])
        bottom = min([min(page.rect.y1,cr.y1+page.rect.height*.35)] + [r.y0-1 for r in aligned if r.y0>=cr.y1])
        direction_candidates = []
        for direction in (('below','above') if is_table else ('above',)):
            low,high = (cr.y1+1,bottom) if direction=='below' else (top,cr.y0-1)
            eligible=[]
            for r in graphics:
                if max(r.width,r.height)<2 or r.width<0 or r.height<0:
                    continue
                if r.y0<low or r.y1>high or min(r.x1,cr.x1)-max(r.x0,cr.x0)<-1:
                    continue
                # Page/column rules cannot seed a scientific visual.
                if r.height<1 and r.width>=page.rect.width*.60 and r.y0<page.rect.height*.08:
                    continue
                if r.height>page.rect.height*.65:
                    continue
                eligible.append(r)
            # Separate unrelated graphics vertically. Keep touching zero-area
            # border rules, which locate the complete table, including headers.
            groups=[]
            group_max_y1 = None
            for r in sorted(eligible,key=lambda r:r.y0):
                if not groups or r.y0-group_max_y1>32:
                    groups.append([r])
                    group_max_y1 = r.y1
                else:
                    groups[-1].append(r)
                    group_max_y1 = max(group_max_y1, r.y1)
            meaningful=[g for g in groups if _bounds(g).width>=45 and _bounds(g).height>=25]
            if not meaningful:
                continue
            group=min(meaningful,key=lambda g: (_bounds(g).y0-cr.y1) if direction=='below' else (cr.y0-_bounds(g).y1))
            clip=_bounds(group)
            gap=clip.y0-cr.y1 if direction=='below' else cr.y0-clip.y1
            if gap>65:
                continue
            # Include nearby axis labels/ticks but never a caption or an entire
            # long prose block. Only one expansion relative to graphics hull.
            halo=fitz.Rect(clip.x0-24,clip.y0-18,clip.x1+24,clip.y1+18)
            label_rects=[]
            for rect,text in lines:
                if (rect.y0>=low and rect.y1<=high and rect.x0>=halo.x0 and rect.x1<=halo.x1
                        and rect.y0>=halo.y0 and rect.y1<=halo.y1 and len(text.strip())<=115
                        and not CAPTION.match(text)):
                    label_rects.append(rect)
            clip=_bounds(group+label_rects)
            clip=fitz.Rect(max(page.rect.x0,clip.x0-2),max(low,clip.y0-2),
                           min(page.rect.x1,clip.x1+2),min(high,clip.y1+2))
            clip, completed_lines = _complete_boundary_lines(clip, lines, low=low, high=high, page_rect=page.rect)
            if clip.width<45 or clip.height<35 or clip.get_area()>page.rect.get_area()*.75:
                continue
            direction_candidates.append((gap,direction,clip,low,high,len(group),completed_lines))
            if direction=='below' and not nearest_table:
                break  # standard table direction preferred; above is fallback
        if not direction_candidates:
            continue
        if nearest_table:
            direction_candidates = [candidate for candidate in direction_candidates
                if not any((candidate[2] & fitz.Rect(previous['bbox'])).get_area()
                    / max(1, candidate[2].get_area()) > .85 for previous in result)]
            if not direction_candidates:
                continue
        gap,direction,clip,low,high,count,completed_lines=min(direction_candidates,key=lambda v:v[0])
        if any((clip & fitz.Rect(c['bbox'])).get_area()/max(1,clip.get_area())>.85 for c in result):
            continue
        result.append(dict(kind='table' if is_table else 'figure',caption=' '.join(caption[4].split()),
                           bbox=list(clip),caption_bbox=list(cr),direction=direction,
                           boundary_lines_completed=completed_lines,
                           crop_policy='caption_guided_graphics_crop',
                           zone_y=[low,high],graphics_count=count,
                           extraction='native_pdf_graphics_crop'))
    return result


def recrop_native_assets(document, output):
    """Re-crop old native assets without altering selected IDs/caption semantics.

    Original asset files and MinerU assets are untouched. Existing caption/page
    identity must match exactly and uniquely. Returns an explicit audit record.
    """
    import fitz
    from PIL import Image
    output=writable_path(Path(output));output.mkdir(parents=True,exist_ok=True)
    audit=[]
    with fitz.open(document.pdf_path) as pdf:
        by_page={}
        for index,asset in enumerate(document.assets):
            if asset.get('extraction')!='native_pdf_caption_graphics_crop':
                continue
            pageno=asset['page']
            if pageno not in by_page:
                by_page[pageno]=caption_graphics_crops(pdf[pageno-1])
            matches=[c for c in by_page[pageno] if c['kind']==asset['kind'] and c['caption']==asset['caption']]
            if len(matches)!=1:
                audit.append(dict(asset_id=asset['id'],status='retained_old_crop_unresolved',page=pageno,matches=len(matches)))
                continue
            crop=matches[0]
            path=writable_path(output/f'page{pageno}_asset{index}.png')
            if path.exists():
                raise FileExistsError('Refusing to overwrite an existing audited crop')
            pdf[pageno-1].get_pixmap(matrix=fitz.Matrix(3,3),clip=fitz.Rect(crop['bbox']),alpha=False).save(path)
            old=deepcopy(asset)
            asset.update(crop,path=str(path),source_crop=dict(pdf_sha256=document.pdf_sha256,page=pageno,
                                                          bbox=crop['bbox'],caption_bbox=crop['caption_bbox']))
            with Image.open(path) as im:
                asset['pixel_width'],asset['pixel_height']=im.size
            audit.append(dict(asset_id=asset['id'],status='recropped',page=pageno,
                              original=old,current=deepcopy(asset)))
    return audit
