from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

from dagent.core.layout.geometry import intersection, text_height, wrap_lines, font


def verify(layout, cfg):
    from PIL import Image
    defects=[]
    def add(category,target,description,severity='hard'):
        defects.append(dict(category=category,target=target,description=description,severity=severity))
    ids=set()
    for e in layout.elements:
        if e.id in ids:
            add('duplicate_id',e.id,'Element IDs must be unique')
        ids.add(e.id)
        box=(e.x,e.y,e.width,e.height)
        if not all(math.isfinite(v) for v in box) or min(e.width,e.height)<=0:
            add('bbox',e.id,'Invalid/non-finite bounding box')
            continue
        if e.x < -.001 or e.y < -.001 or e.x+e.width > layout.width+.001 or e.y+e.height > layout.height+.001:
            add('clipping',e.id,'Element extends outside canvas')
        if e.x < cfg.margin_inches-.01 or e.y < cfg.margin_inches-.01 or e.x+e.width > layout.width-cfg.margin_inches+.01 or e.y+e.height > layout.height-cfg.margin_inches+.01:
            add('margin',e.id,'Element violates safe print margin')
        if e.text:
            if not math.isfinite(e.font_pt) or e.font_pt<cfg.minimum_font_pt:
                add('font',e.id,'Font below readable minimum')
                continue
            needed=text_height(e.text,e.width,e.font_pt,e.bold,e.line_spacing)
            if needed>e.height+.02:
                add('text_overflow',e.id,f'Estimated required height {needed:.3f} > {e.height:.3f}')
            max_width=(e.width-.16)*72/e.font_pt*100
            if any(font(e.bold).getlength(line)>max_width for line in wrap_lines(e.text,e.width,e.font_pt,e.bold)):
                add('long_token',e.id,'Unbreakable token wider than textbox')
        if e.kind=='figure':
            if not e.asset_path or not Path(e.asset_path).is_file():
                add('missing_image',e.id,'Missing image file')
                continue
            with Image.open(e.asset_path) as im:
                dpi=min(im.width/e.width,im.height/e.height)
                if dpi<cfg.minimum_image_dpi:
                    add('image_resolution',e.id,f'Effective image resolution {dpi:.1f} DPI below threshold')
                if abs((e.width/e.height)/(im.width/im.height)-1)>.04:
                    add('image_distortion',e.id,'Figure aspect ratio distorted')
    for i,a in enumerate(layout.elements):
        for b in layout.elements[i+1:]:
            if intersection(a,b)>.01:
                add('overlap',a.id,f'Overlaps {b.id}')
    if cfg.design == 'editorial' and cfg.editorial_flow == 'bands':
        headings = sorted((e for e in layout.elements if e.kind=='heading'),key=lambda e:e.y)
        sids = [e.section_id for e in headings]
        if len(sids)!=len(set(sids)):
            add('section_topology','bands','Exactly one heading per scientific section')
        for index, head in enumerate(headings):
            items = [e for e in layout.elements if e.section_id==head.section_id and
                     e.kind in ('text','figure','caption','table_cell','table_note')]
            ceiling = headings[index+1].y if index+1<len(headings) else layout.height-cfg.margin_inches
            if not items:
                add('orphan_heading',head.id,'Band heading must have its own content')
            if abs(head.x-cfg.margin_inches)>.01 or abs(head.width-(layout.width-2*cfg.margin_inches))>.01:
                add('section_topology',head.id,'Band heading must span the safe page width')
            for e in items:
                if e.y<head.y+head.height-.01 or e.y+e.height>ceiling+.01:
                    add('section_topology',e.id,'Band content must remain between its own and the next heading')
                if e.kind=='figure':
                    caption = layout.get(e.id.removesuffix('_image')+'_caption')
                    if (caption is None or caption.section_id!=head.section_id or
                            caption.y<e.y+e.height-.01 or caption.y-e.y-e.height>cfg.gap_inches+.1 or
                            caption.x>e.x+.01 or caption.x+caption.width<e.x+e.width-.01):
                        add('detached_caption',e.id,'Whole figure must be followed by its own caption within the evidence lane')
        for e in layout.elements:
            if e.kind in ('text','figure','caption','table_cell','table_note') and e.section_id not in sids:
                add('section_topology',e.id,'Every body element needs a scientific band heading')
    elif cfg.design in ('balanced', 'portfolio', 'editorial') and any(e.kind == 'heading' for e in layout.elements):
        cw=(layout.width-2*cfg.margin_inches-(cfg.columns-1)*cfg.gap_inches)/cfg.columns
        starts=sorted({round(e.x, 3) for e in layout.elements if e.kind=='heading'})
        def column(e):
            if cfg.design in ('portfolio', 'editorial'):
                return max(0, sum(e.x+e.width/2 >= x for x in starts)-1)
            return min(cfg.columns-1,max(0,int((e.x+e.width/2-cfg.margin_inches)/(cw+cfg.gap_inches))))
        for col in range(len(starts) if cfg.design in ('portfolio', 'editorial') else cfg.columns):
            items=sorted((e for e in layout.elements if e.kind not in ('title','authors','takeaway','takeaway_heading','rule','footer_heading','footer_text') and column(e)==col),key=lambda e:e.y)
            active=None
            for i,e in enumerate(items):
                if e.kind=='heading':
                    active=e.section_id
                    if i+1==len(items) or items[i+1].kind=='heading' or items[i+1].section_id!=active:
                        add('orphan_heading',e.id,'Heading must precede content of its own section in the same column')
                elif e.section_id!=active:
                    add('section_topology',e.id,'Content must remain under its own section heading')
                if e.kind=='figure':
                    caption=layout.get(e.id.removesuffix('_image')+'_caption')
                    if caption and (i+1==len(items) or items[i+1].id!=caption.id):
                        add('detached_caption',e.id,'Figure and caption must remain adjacent in one column')
    # A full-width takeaway is above the columns, not an unheaded column block.
    takeaways=[e for e in layout.elements if e.kind=='takeaway']
    if takeaways:
        heads=[e for e in layout.elements if e.kind=='takeaway_heading']
        if len(heads)!=1:
            add('takeaway_topology','takeaway','Exactly one explicit takeaway heading required')
        column_tops=[e.y for e in layout.elements if e.kind=='heading']
        for e in takeaways:
            if e.section_id!='conclusion' or not e.source_ids or e.asset_path:
                add('takeaway_source',e.id,'Takeaway must preserve a source-grounded conclusion')
            if abs(e.x-cfg.margin_inches)>.01 or abs(e.width-(layout.width-2*cfg.margin_inches))>.01:
                add('takeaway_topology',e.id,'Takeaway must span the safe full page width')
            if heads and heads[0].y+heads[0].height>e.y+.01:
                add('takeaway_topology',e.id,'Takeaway must follow its own heading')
            if column_tops and e.y+e.height>min(column_tops)+.01:
                add('takeaway_topology',e.id,'Takeaway must be above every body column')
    footers=[e for e in layout.elements if e.kind=='footer_text']
    if footers:
        heads=[e for e in layout.elements if e.kind=='footer_heading']
        if len(heads)!=1:
            add('footer_topology','footer','Exactly one footer heading required')
        body=[e for e in layout.elements if e.kind in ('text','figure','caption','heading','table_cell','table_note')]
        for e in footers:
            if e.section_id!='conclusion' or not e.source_ids or e.asset_path:
                add('footer_source',e.id,'Footer must preserve source-grounded conclusion text')
            if abs(e.x-cfg.margin_inches)>.01 or abs(e.width-(layout.width-2*cfg.margin_inches))>.01:
                add('footer_topology',e.id,'Footer must span the safe full page width')
            if heads and heads[0].y+heads[0].height>e.y+.01:
                add('footer_topology',e.id,'Footer text must follow its own heading')
        if heads and body and max(e.y+e.height for e in body)>heads[0].y+.01:
            add('footer_topology','footer','Footer must be below all body columns')
    text_elements=[e for e in layout.elements if e.kind in ('text','heading','caption')]
    aligned=sum(any(abs(e.x-other.x)<.08 for other in text_elements if other.id!=e.id) for e in text_elements)/max(1,len(text_elements))
    occupied=sum(e.width*e.height for e in layout.elements)/(layout.width*layout.height)
    fonts=[e.font_pt for e in layout.elements if e.kind=='text']
    readable=min(1,min(fonts,default=cfg.body_font_pt)/cfg.body_font_pt)
    whitespace=max(0,1-occupied)
    whitespace_score=max(0,1-abs(whitespace-.28)/.72)
    hard=sum(d['severity']=='hard' for d in defects)
    score=max(0, .4*readable+.35*aligned+.25*whitespace_score-min(1,hard*.16))
    return dict(valid=not hard, hard_violations=hard, defects=defects, score=score,
                readability_proxy=readable,alignment=aligned,whitespace_fraction=whitespace,
                overflow_check='font-metric estimate; final PDF glyph bounds checked separately')
